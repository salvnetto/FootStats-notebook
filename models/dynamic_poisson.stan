data {
  //train
  int<lower=1> nrounds;
  int<lower=1> nteams;
  int<lower=1> ngames;
  array[ngames] int<lower=1, upper=nrounds> round_t;
  array[ngames] int<lower=1, upper=nteams> team1; // Home team index
  array[ngames] int<lower=1, upper=nteams> team2; // Away team index
  array[ngames] int<lower=0> gf;        // Goals scored by the home team
  array[ngames] int<lower=0> ga;        // Goals scored by the away team

  //test
  int<lower=1> ngames_new;
  array[ngames_new] int<lower=1, upper=nrounds> round_t_new;
  array[ngames_new] int<lower=1, upper=nteams> team1_new;
  array[ngames_new] int<lower=1, upper=nteams> team2_new;
}

parameters {
  matrix[nrounds, nteams] att_raw;
  matrix[nrounds, nteams] def_raw;
  real home;
  real<lower=0> sigma_att;
  real<lower=0> sigma_def;
}

transformed parameters {
  matrix[nrounds, nteams] att;
  matrix[nrounds, nteams] def;
  vector[ngames] theta1;
  vector[ngames] theta2;

  for (t in 1:nrounds){
    att[t] = att_raw[t] - mean(att_raw[t]);
    def[t] = def_raw[t] - mean(def_raw[t]);
  }


  for (k in 1:ngames) {
    theta1[k] = att[round_t[k], team1[k]] - def[round_t[k], team2[k]] + home;
    theta2[k] = att[round_t[k], team2[k]] - def[round_t[k], team1[k]];
  }
}

model{
  // Priors
  sigma_att ~ cauchy(0, 25);
  sigma_def ~ cauchy(0, 25);

  home ~ normal(0, 10);

  target += normal_lpdf(att_raw[1] | 0, sigma_att);
  target += normal_lpdf(def_raw[1] | 0, sigma_def);


  for (t in 2:nrounds) {
      target += normal_lpdf(att_raw[t] | att_raw[t-1], sigma_att);
      target += normal_lpdf(def_raw[t] | def_raw[t-1], sigma_def);
  }

  // Likelihood
  for (k in 1:ngames) {
    target += poisson_log_lpmf(gf[k] | theta1[k]);
    target += poisson_log_lpmf(ga[k] | theta2[k]);
  }
}

generated quantities {
  vector[ngames] log_lik;
  vector[ngames_new] theta1_new;
  vector[ngames_new] theta2_new;
  array[ngames_new] int gf_new;
  array[ngames_new] int ga_new;

  // Log-likelihood for training data
  for (k in 1:ngames) {
    log_lik[k] = poisson_log_lpmf(gf[k] | theta1[k]) +
                 poisson_log_lpmf(ga[k] | theta2[k]);
  }

  // Predictive distributions for test data
  for (k in 1:ngames_new) {
    theta1_new[k] = att[round_t_new[k], team1_new[k]] - def[round_t_new[k], team2_new[k]] + home;
    theta2_new[k] = att[round_t_new[k], team2_new[k]] - def[round_t_new[k], team1_new[k]];
  }

  gf_new = poisson_log_rng(theta1_new);
  ga_new = poisson_log_rng(theta2_new);
}
