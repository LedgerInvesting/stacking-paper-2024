functions {
  vector gamma_pars(real mean, real variance) {
      real shape = pow(mean, 2) / variance;
      real scale = variance / mean;
      return [shape, scale]';
  }
      
  real gamma2_lpdf(real y, real mean, real variance) {
      vector[2] pars = gamma_pars(mean, variance);
      return gamma_lpdf(y | pars[1], 1 / pars[2]);
  }
  real gamma2_rng(real mean, real variance) {
      vector[2] pars = gamma_pars(mean, variance);
      return gamma_rng(pars[1], 1 / pars[2]);
  }
}

data {
  int<lower=0> N;
  int<lower=0> T_train;
  int<lower=0> T_test;
  array[N] vector<lower=0>[T_train] AY_train;
  array[N] vector<lower=0>[T_test] AY_test;
  array[N] vector<lower=0>[T_train] loss_ratio_train;
  array[N] vector<lower=0>[T_test] loss_ratio_test;
}

transformed data {
  array[N] vector[T_train] AY_train_z;
  array[N] vector[T_test] AY_test_z;

  for(i in 1:N){
    AY_train_z[i] = AY_train[i] - max(AY_train[i]) / 2.0;
    AY_test_z[i] = AY_test[i] - max(AY_test[i]) / 2.0;
  }
}

parameters {
  real alpha;
  real beta;
  cholesky_factor_corr[3] L_omega;
  matrix[3, N] Z;
  vector<lower=0>[3] sigma_u;
  real delta;
}

transformed parameters {
  matrix[N, T_train] lp;
  matrix[N, 3] u = (diag_pre_multiply(sigma_u, L_omega) * Z)';
  vector[N] sigma = delta + u[,3];

  for(i in 1:N){
    for(j in 1:T_train){
      real mu = alpha + u[i, 1] + (beta + u[i, 2]) * AY_train_z[i][j];
      lp[i][j] = gamma2_lpdf(loss_ratio_train[i][j] | exp(mu), exp(sigma[i]));
    }
  }
}

model {
  alpha ~ std_normal();
  beta ~ std_normal();
  L_omega ~ lkj_corr_cholesky(2);
  to_vector(Z) ~ std_normal();
  sigma_u ~ std_normal();
  delta ~ std_normal();
  target += sum(lp);
}

generated quantities {
  corr_matrix[3] Omega = L_omega * L_omega';
  array[N] vector[T_test] log_lik;
  array[N] vector<lower=0>[T_test] post_pred;

  for(i in 1:N){
    for(j in 1:T_test){
      real mu = alpha + u[i, 1] + (beta + u[i, 2]) * AY_test_z[i][j];
      log_lik[i][j] = gamma2_lpdf(loss_ratio_test[i][j] | exp(mu), exp(sigma[i]));
      post_pred[i][j] = gamma2_rng(exp(mu), exp(sigma[i]));
    }
  }
}
