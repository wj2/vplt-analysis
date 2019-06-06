
data {
//sizes 
  int<lower=0> N; // number of samples
  int<lower=0> K; // number of outcomes
  int<lower=0> L; // number of images
  
  // prior data
  real prior_eps_mean;
  real<lower=0> prior_eps_var;
  real<lower=0> prior_bias_var;
  real<lower=0> prior_salience_var;

  // main data
  int imgs[N, L, K] ; // image selection matrix
  matrix[N, K] novs; // novel indicator
  matrix[N, K] views; // number of previous views
  int y[N]; // outcome vector
}

parameters {
  vector[L] s; // image inherent saliences
  real<lower=0> salience_var;
  vector<lower=0>[K - 1] bias; // target bias terms
  real eps; // novelty bias
}

model {
  // var declarations
  matrix[N, K] img_s;
  matrix[N, K] outcome_evidence;

  // priors
  eps ~ normal(prior_eps_mean, prior_eps_var);
  bias ~ normal(0, prior_bias_var);
  // salience_var ~ normal(0, prior_salience_var);
  s ~ normal(0, salience_var);
  
  // setup
  for (k in 1:K) {
    img_s[:, k] = to_matrix(imgs[:, :, k])*s + bias[k];
  }
  
  outcome_evidence = img_s + novs * eps;
    
  for (n in 1:N) {
    y[n] ~ categorical_logit(outcome_evidence[n]');
  }
}
