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

parameters {
  real alpha;
  real phi_star;
  vector[N] init;
  cholesky_factor_corr[3] L_omega;
  matrix[3, N] Z;
  vector<lower=0>[3] sigma_u;
  real delta;
}

transformed parameters {
  matrix[N, T_train] lp;
  matrix[N, 3] u = (diag_pre_multiply(sigma_u, L_omega) * Z)';
  vector[N] phi = inv_logit(phi_star + u[, 2]) * 2 - 1;
  vector[N] sigma = delta + u[, 3];

  for(i in 1:N){
    for(j in 1:T_train){
      real lag = (j == 1) ? 0 : log(loss_ratio_train[i][j - 1]);
      real mu = (j == 1) ? init[i] : (1 - phi[i]) * (alpha + u[i, 1]) + phi[i] * lag;
      lp[i][j] = gamma2_lpdf(loss_ratio_train[i][j] | exp(mu), exp(sigma[i]));
    }
  }
}

model {
  alpha ~ std_normal();
  phi_star ~ std_normal();
  init ~ std_normal();
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
      real lag = j == 1 ? log(loss_ratio_train[i][T_train]) : log(loss_ratio_test[i][j - 1]);
      real mu = (1 - phi[i]) * (alpha + u[i, 1]) + phi[i] * lag;
      log_lik[i][j] = gamma2_lpdf(loss_ratio_test[i][j] | exp(mu), exp(sigma[i]));
      post_pred[i][j] = gamma2_rng(exp(mu), exp(sigma[i]));
    }
  }
}
