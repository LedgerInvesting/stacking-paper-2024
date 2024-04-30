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
  array[N] vector<lower=0>[T_train] loss_ratio_train;
  array[N] vector<lower=0>[T_test] loss_ratio_test;
}

parameters {
  real alpha;
  vector[N] u_z;
  real<lower=0> sigma_u;
  real<lower=0> sigma;
}

transformed parameters {
  matrix[N, T_train] lp;
  vector[N] u = sigma_u * u_z;

  for(i in 1:N){
    for(j in 1:T_train){
      real mu = alpha + u[i];
      lp[i][j] = gamma2_lpdf(loss_ratio_train[i][j] | exp(mu), sigma);
    }
  }
}

model {
  alpha ~ std_normal();
  u_z ~ std_normal();
  sigma_u ~ std_normal();
  sigma ~ std_normal();
  target += sum(lp);
}

generated quantities {
  array[N] vector[T_test] log_lik;
  array[N] vector<lower=0>[T_test] post_pred;

  for(i in 1:N){
    for(j in 1:T_test){
      real mu = alpha + u[i];
      log_lik[i][j] = gamma2_lpdf(loss_ratio_test[i][j] | exp(mu), sigma);
      post_pred[i][j] = gamma2_rng(exp(mu), sigma);
    }
  }
}
