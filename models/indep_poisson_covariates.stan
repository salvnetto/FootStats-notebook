data {
  //train
  int<lower=1> nteams;
  int<lower=1> ngames;
  array[ngames] int team1;
  array[ngames] int team2;
  array[ngames] int<lower=0> gf;
  array[ngames] int<lower=0> ga;
  int<lower=1> ncol_X;
  matrix[ngames, ncol_X] X;
  //test
  int<lower=1> ngames_new;
  array[ngames_new] int team1_new;
  array[ngames_new] int team2_new;
  int<lower=1> ncol_Xnew;
  matrix[ngames_new, ncol_Xnew] Xnew;
}

parameters {
  sum_to_zero_vector[nteams] att;
  sum_to_zero_vector[nteams] def;
  real home;
  real mu_att;
  real mu_def;
  real<lower=0> sd_att;
  real<lower=0> sd_def;
  vector[ncol_X] beta;
}

transformed parameters {
  vector[ngames] theta1;
  vector[ngames] theta2;

  theta1 = (att[team1] - def[team2] + home + X*beta);
  theta2 = (att[team2] - def[team1] + X*beta);
}

model {
  // Priors
  att ~ normal(0, sd_att);
  def ~ normal(0, sd_def);
  home ~ normal(0, 10);
  beta ~ normal(0, 10);
  mu_att ~ normal(0, 10);
  mu_def ~ normal(0, 10);
  sd_att ~ cauchy(0, 2.5);
  sd_def ~ cauchy(0, 2.5);

  // Likelihood
  for (k in 1:ngames) {
    target += poisson_log_lpmf(gf[k] | theta1[k]);
    target += poisson_log_lpmf(ga[k] | theta2[k]);
  }
}

generated quantities {
  vector[ngames] log_lik;         // Log-likelihood for model comparison
  vector[ngames_new] theta1_new;  // Expected goals for new home teams
  vector[ngames_new] theta2_new;  // Expected goals for new away teams
  array[ngames_new] int gf_new;
  array[ngames_new] int ga_new;

  // Log-likelihood for training data
  for (k in 1:ngames) {
    log_lik[k] = poisson_log_lpmf(gf[k] | theta1[k]) +
                 poisson_log_lpmf(ga[k] | theta2[k]);
  }

  // Predictive distributions for test data
  theta1_new = (att[team1_new] - def[team2_new] + home + Xnew*beta);
  theta2_new = (att[team2_new] - def[team1_new] + Xnew*beta);

  gf_new = poisson_log_rng(theta1_new);
  ga_new = poisson_log_rng(theta2_new);
}
