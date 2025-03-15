data {
  //train
  int<lower=1> nteams;
  int<lower=1> ngames;
  array[ngames] int team1;
  array[ngames] int team2;
  array[ngames] int<lower=0> gf;
  array[ngames] int<lower=0> ga;
  //test
  int<lower=1> ngames_new;
  array[ngames_new] int team1_new;
  array[ngames_new] int team2_new;
}

parameters {
  vector[nteams] att_h;
  vector[nteams] att_a;
  vector[nteams] def_h;
  vector[nteams] def_a;
}

transformed parameters {
  vector[ngames] theta1;
  vector[ngames] theta2;

  theta1 = (att_h[team1]*def_a[team2]);
  theta2 = (att_a[team2]*def_h[team1]);
}

model {
  // Priors
  att_h ~ normal(0, 10);
  att_a ~ normal(0, 10);
  def_h ~ normal(0, 10);
  def_a ~ normal(0, 10);

  // Likelihood
  for (k in 1:ngames) {
    target += poisson_lpmf(gf[k] | theta1[k]);
    target += poisson_lpmf(ga[k] | theta2[k]);
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
    log_lik[k] = poisson_lpmf(gf[k] | theta1[k]) +
                 poisson_lpmf(ga[k] | theta2[k]);
  }

  // Predictive distributions for test data
  theta1 = (att_h[team1_new]*def_a[team2_new]);
  theta2 = (att_a[team2_new]*def_h[team1_new]);

  gf_new = poisson_rng(theta1_new);
  ga_new = poisson_rng(theta2_new);
}
