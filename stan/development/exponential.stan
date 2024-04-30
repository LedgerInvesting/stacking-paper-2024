functions {
    real gamma_mean_variance_lpdf(real y, real mu, real sigma2) {
        real shape;
        real inv_scale;
        real mu_pr;
        real sigma2_pr;
        mu_pr = mu <= 0.00001 ? 0.00001: mu;
        sigma2_pr = sigma2 <= 0.00001 ? 0.00001: sigma2;
        shape = pow(mu_pr, 2) / sigma2_pr;
        inv_scale = mu_pr / sigma2_pr;
        return gamma_lpdf(y | shape, inv_scale);
    }

    real gamma_mean_variance_rng(real mu, real sigma2) {
        real shape;
        real inv_scale;
        real mu_pr;
        real sigma2_pr;
        mu_pr = mu <= 0.00001 ? 0.00001: mu;
        sigma2_pr = sigma2 <= 0.00001 ? 0.00001: sigma2;
        shape = pow(mu_pr, 2) / sigma2_pr;
        inv_scale = mu_pr / sigma2_pr;
        return gamma_rng(shape, inv_scale);
    }
}

data {
  int N;
  int DL_start;
  int AY_train;
  int DL_train;
  int AY_test;
  int DL_test;
  int AY_valid;
  int DL_valid;
  array[N] matrix[AY_train, DL_train] loss_ratio_train;
  array[N] matrix[AY_test, DL_test] loss_ratio_test;
  array[N] matrix[AY_valid, DL_valid] loss_ratio_valid;
}

parameters {
  real beta0_mu;
  real<lower=0> beta0_sigma;

  real beta1_mu;
  real<lower=0> beta1_sigma;

  real sigma0;
  real<lower=0> sigma1;
  
  vector[N] beta0_pr;
  vector[N] beta1_pr;
}

transformed parameters {
  vector[N] beta0 = beta0_mu + beta0_sigma * beta0_pr;  
  vector<lower=0>[N] beta1 = exp(beta1_mu + beta1_sigma * beta1_pr);
}

model {
  sigma0 ~ normal(-4, 1);
  sigma1 ~ normal(0.5, 0.25);

  beta0_mu ~ normal(1, 1);
  beta1_mu ~ normal(-.5, 1);
  
  beta0_sigma ~ normal(0, 1);
  beta1_sigma ~ normal(0, 1);

  beta0_pr ~ std_normal();
  beta1_pr ~ std_normal();

  {
    real mu;
    real sigma2;
    real ata;
    for (i in 1:N) {
      for (acc_year in 1:AY_train) {        
        for (dev_lag in 2:DL_train) {
          if (dev_lag >= DL_start) {
            ata = 1 + exp(beta0[i] - beta1[i] * (dev_lag-1));
            mu = loss_ratio_train[i][acc_year, dev_lag-1] * ata;
            sigma2 = pow(exp(sigma0 - sigma1 * (dev_lag-1)), 2);
            target += gamma_mean_variance_lpdf(
              loss_ratio_train[i][acc_year, dev_lag] | mu, sigma2
            );      
          }
        }
      }     
    }
  }
}

generated quantities {
  array[N] matrix[AY_train, DL_train-1] log_lik_train;
  array[N] matrix[AY_train, DL_train-1] post_pred_train;
  array[N] matrix[AY_test, DL_test] log_lik_test;
  array[N] matrix[AY_test, DL_test] post_pred_test;
  array[N] matrix[AY_valid, DL_valid] log_lik_valid;
  array[N] matrix[AY_valid, DL_valid] post_pred_valid;
  
  {
    real mu;
    real sigma2;
    real ata;
    int dl_offset;
    for (i in 1:N) {
      for (acc_year in 1:AY_train) {
        for (dev_lag in 2:DL_train) {
          ata = 1 + exp(beta0[i] - beta1[i] * (dev_lag-1));
          mu = loss_ratio_train[i][acc_year, dev_lag-1] * ata;
          sigma2 = pow(exp(sigma0 - sigma1 * (dev_lag-1)), 2);
          log_lik_train[i][acc_year,dev_lag-1] = gamma_mean_variance_lpdf(
            loss_ratio_train[i][acc_year, dev_lag] | mu, sigma2
          );
          post_pred_train[i][acc_year,dev_lag-1] = gamma_mean_variance_rng(mu, sigma2);     
        }
      }
      for (acc_year in 1:AY_test) {
        for (dev_lag in 1:DL_test) {
          dl_offset = DL_train;
          ata = 1 + exp(beta0[i] - beta1[i] * (dev_lag+dl_offset-1));
          if (dev_lag == 1) {
            mu = loss_ratio_train[i][acc_year, DL_train] * ata;
          } else {
            mu = post_pred_test[i][acc_year, dev_lag-1] * ata;
          }
          sigma2 = pow(exp(sigma0 - sigma1 * (dev_lag+dl_offset-1)), 2);
          log_lik_test[i][acc_year,dev_lag] = gamma_mean_variance_lpdf(
            loss_ratio_test[i][acc_year, dev_lag] | mu, sigma2
          );
          post_pred_test[i][acc_year,dev_lag] = gamma_mean_variance_rng(mu, sigma2);     
        }
      }
      for (acc_year in 1:AY_valid) {
        for (dev_lag in 1:DL_valid) {
          dl_offset = DL_train + DL_test;
          ata = 1 + exp(beta0[i] - beta1[i] * (dev_lag+dl_offset-1));
          if (dev_lag == 1) {
            mu = post_pred_test[i][acc_year, DL_test] * ata;
          } else {
            mu = post_pred_valid[i][acc_year, dev_lag-1] * ata;
          }
          sigma2 = pow(exp(sigma0 - sigma1 * (dev_lag+dl_offset-1)), 2);
          log_lik_valid[i][acc_year,dev_lag] = gamma_mean_variance_lpdf(
            loss_ratio_valid[i][acc_year, dev_lag] | mu, sigma2
          );
          post_pred_valid[i][acc_year,dev_lag] = gamma_mean_variance_rng(mu, sigma2);     
        }
      }
    }
  }
}