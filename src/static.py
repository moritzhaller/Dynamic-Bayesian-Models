import numpy as np
import pandas as pd
import pymc3 as pm, theano.tensor as tt
from pymc3.backends import Text
import matplotlib.pyplot as plt
import re

# Import season 2013/14 fixtures
df = pd.read_csv('../data/ts14-15.csv')
df = df.drop('Unnamed: 0', 1)

# Build team index
teams = df.home_team.unique()
teams = pd.DataFrame(teams, columns=['team'])
teams['i'] = teams.index

df = pd.merge(df, teams, left_on='home_team', right_on='team', how='left')
df = df.rename(columns = {'i': 'i_home'}).drop('team', 1)
df = pd.merge(df, teams, left_on='away_team', right_on='team', how='left')
df = df.rename(columns = {'i': 'i_away'}).drop('team', 1)

df = df.sort_values(by='kick_off', ascending=1)

# Observed goals stats (Eyeball Poisson)
observed_home_goals = df.home_score.values
observed_away_goals = df.away_score.values

home_team = df.i_home.values
away_team = df.i_away.values

num_teams = len(df.i_home.drop_duplicates())
num_games = len(home_team)

# Add back to back fixture round index t
fixtures_per_round = num_teams/2
num_rounds = 38

rounds = np.array([[x]*(fixtures_per_round) for x in range(1,39)]).flatten()
df['t'] = 1
df['t'] = rounds

T = observed_home_goals.shape[0]

trace_len = 10000
num_weeks = 38

# If run with T == 1, only fit timestep 1 without AR
for t in range(1,num_weeks+1):
    print "\nTrain week %d of %d (i=%d)" %(t, num_weeks, t)

    with pm.Model() as exp_1:    
        # global model parameters
        home = pm.Normal('home', mu=0, tau=0.0001)
        tau_att = pm.Gamma('tau_att', alpha=0.1, beta=0.1)
        tau_def = pm.Gamma('tau_def', alpha=0.1, beta=0.1)
        intercept = pm.Normal('intercept', mu=0, tau=0.0001)

        atts_star   = pm.Normal("atts_star",
                               mu = 0,
                               tau = tau_att,
                               shape = num_teams)
        defs_star   = pm.Normal("defs_star",
                               mu = 0,
                               tau = tau_def,
                               shape = num_teams)
        # Identifieability
        atts = pm.Deterministic('atts', atts_star - tt.mean(atts_star))
        defs = pm.Deterministic('defs', defs_star - tt.mean(defs_star))
        home_theta  = tt.exp(intercept + home + atts[home_team[:t*10]] + defs[away_team[:t*10]])
        away_theta  = tt.exp(intercept + atts[away_team[:t*10]] + defs[home_team[:t*10]])

        # Likelihood
        home_points = pm.Poisson('home_points', mu=home_theta, observed=observed_home_goals[:t*10])
        away_points = pm.Poisson('away_points', mu=away_theta, observed=observed_away_goals[:t*10])
        
        # Sampling
        # If no previous model runs, find map as starting point
        # otherwise use estimates from previous model run
        # No starting point for atts_ni{t+1} and defs respectively, could use atts_ni{t} estimates instead
        if (t == 1):
            start = pm.find_MAP()
        else:
            tracename = "trace_exp_1_nuts_" + str(t-1)
            
            print "Load last trace '" + tracename + "'"
            
            trace_loaded = pm.backends.text.load(tracename)
            pattern = re.compile("(atts\d|defs\d|.*_ni" + str(t) + ")")
            start = {key: np.mean(trace_loaded[key], axis=0) for key in trace_loaded.varnames if pattern.match(key) == None}
            
        # step = pm.NUTS()
        step = pm.NUTS()
        db = Text("trace_exp_1_nuts_{0}".format(t))
        trace = pm.sample(trace_len, step, start=start, trace=db)