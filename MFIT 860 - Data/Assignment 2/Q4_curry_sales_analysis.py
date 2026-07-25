"""
Q4 - Curry Sales Analysis
MMA 860 Assignment 2

Structure (matches the cheat sheet workflow: F-test -> t-tests -> diagnostics -> R^2):
  Part a) Build a regression model to predict Sales
  Part b) One-tailed t-test: does the US respond MORE to Ad_Budget than Canada?
  Part c) Chow test: is the whole Sales relationship structurally different for US vs Canada?

This script uses statsmodels throughout (Cheat Sheets 5/6, 8, 9) since it gives us the
F-test, t-tests, and .wald_test() we need for hypothesis testing in one place.
"""

import pandas as pd
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

# ---------------------------------------------------------------------------
# 0. Import the data
# ---------------------------------------------------------------------------
data_path = osp.join(osp.curdir, '860 Assignment 2 Data.xlsx')
data = pd.read_excel(data_path, sheet_name='Curry')

print(data.dtypes)
print(data.describe(include='all'))
print(data['Country'].value_counts())   # 138 US, 62 CA in this sample

# ===========================================================================
# PART A - Build a model to predict Sales
# ===========================================================================
# Country is a string, so we can't feed it into the regression as-is.
# Using C(Country) in the formula tells statsmodels/patsy to dummy-code it.
# Patsy dummy-codes alphabetically and drops the first category as the
# reference group, so CA becomes the baseline (0) and "C(Country)[T.US]"
# is a 1/0 flag for US. That's exactly what we want, since the business
# question is framed as "how does the US differ from Canada".
model_a = ols('Sales ~ Ad_Budget + Price + Distance + C(Country)', data).fit()
print(model_a.summary())

# --- Step 1: Does the model have any predictive power? (Joint F-test) -----
print("F-statistic:", model_a.fvalue, "p-value:", model_a.f_pvalue)
# F is enormous and the p-value is ~0, so the model, as a whole, clearly has
# predictive power (we reject the null that all slopes are jointly zero).

# --- Step 2: Do the variables belong in the model? (t-tests) --------------
# Read the P>|t| column in the summary above. Ad_Budget, Price and
# C(Country)[T.US] are all significant at the 5% level. Distance is not
# (p ~ 0.13). We keep Distance anyway: it's central to the business question
# ("how far people live from a retailer"), and dropping a theoretically
# relevant variable just because one t-test misses 0.05 is exactly the kind
# of "R^2-first" mistake the cheat sheet warns against. This is a judgment
# call, not a coding decision - flagging it here rather than hiding it.

# --- Step 3: Regression assumption diagnostics -----------------------------
predicted_y = model_a.fittedvalues
residuals = model_a.resid

fig, axes = plt.subplots(2, 2, figsize=(11, 9))

# Residuals vs Fitted - look for absence of pattern, centered on 0
axes[0, 0].scatter(predicted_y, residuals, s=12, c='black')
axes[0, 0].hlines(0, predicted_y.min(), predicted_y.max(), color='red', linestyles='dashed')
axes[0, 0].set_xlabel("Model Prediction")
axes[0, 0].set_ylabel("Residual")
axes[0, 0].set_title("Residuals vs Fitted")

# Normal Q-Q - look for points hugging the diagonal
stats.probplot(residuals, dist='norm', plot=axes[0, 1])
axes[0, 1].set_title("Normal Q-Q")

# Scale-Location - look for a flat band (homoskedasticity)
norm_resid = (residuals - residuals.mean()) / residuals.std()
axes[1, 0].scatter(predicted_y, np.sqrt(np.abs(norm_resid)), c='black', s=12)
axes[1, 0].set_xlabel("Fitted Values")
axes[1, 0].set_ylabel("Root of standardized residual")
axes[1, 0].set_title("Scale-Location")

# Residuals vs Leverage / Cook's Distance
sm.graphics.influence_plot(model_a, ax=axes[1, 1], criterion="cooks")
axes[1, 1].set_title("Residuals vs Leverage")

plt.tight_layout()
plt.show()

# Quick numeric check for heteroskedasticity (Cheat Sheet 10)
from statsmodels.stats.diagnostic import het_breuschpagan
bp = het_breuschpagan(model_a.resid, model_a.model.exog)
print(dict(zip(('LM Stat', 'LM p-value', 'F Stat', 'F p-value'), bp)))
# p-values well above 0.05 -> fail to reject homoskedasticity, no HC3 correction needed.

# VIF check (Cheat Sheet 5/6 "try it yourself")
from statsmodels.stats.outliers_influence import variance_inflation_factor
X_vif = model_a.model.exog
vif_data = pd.DataFrame({
    'variable': model_a.model.exog_names,
    'VIF': [variance_inflation_factor(X_vif, i) for i in range(X_vif.shape[1])]
})
print(vif_data)
# All VIFs are ~1 -> no multicollinearity problem between Ad_Budget, Price,
# Distance and Country.

# --- Step 4: Is R^2 good enough for the business need? --------------------
print("R-squared:", model_a.rsquared, "Adj R-squared:", model_a.rsquared_adj)
# R^2 ~ 0.99 - Ad_Budget alone is doing most of the work here (it's very
# tightly related to Sales), and the model explains almost all the variance.
# This is more than sufficient for a business forecasting use case.

# ===========================================================================
# PART B - Does the US respond MORE to Ad_Budget than Canada?
# (most powerful test = ONE-TAILED t-test on the interaction term)
# ===========================================================================
# The claim is directional ("more", not just "different"), so instead of a
# two-tailed test (which splits alpha across both tails) we use a one-tailed
# test, which puts the full alpha in the direction we actually care about.
# For the same alpha, a one-tailed test has more power to detect an effect
# in the hypothesized direction than a two-tailed test does - that's why
# it's the "most powerful" choice here, not a Wald/F-test on multiple terms.
#
# We test this using the coefficient on the interaction between Ad_Budget
# and Country. A positive, significant interaction would mean the US slope
# on Ad_Budget is higher than Canada's.
model_b = ols('Sales ~ Ad_Budget + Price + Distance + C(Country) + Ad_Budget:C(Country)', data).fit()
print(model_b.summary())

# H0: beta_(Ad_Budget x US)  <= 0   (US does NOT respond more than Canada)
# Ha: beta_(Ad_Budget x US)   > 0   (US responds MORE than Canada)
interaction_coef = 'Ad_Budget:C(Country)[T.US]'
t_stat = model_b.tvalues[interaction_coef]
p_two_tailed = model_b.pvalues[interaction_coef]

# Convert statsmodels' two-tailed p-value to a one-tailed p-value:
# if t points the direction Ha claims (positive here), halve the p-value;
# otherwise there's no way to reject Ha, so the one-tailed p-value is
# whatever probability mass is left on that side (> 0.5).
p_one_tailed = p_two_tailed / 2 if t_stat > 0 else 1 - p_two_tailed / 2

print("Interaction coefficient:", model_b.params[interaction_coef])
print("t-statistic:", t_stat)
print("One-tailed p-value:", p_one_tailed)
# Result: t is negative and the one-tailed p-value is large (>> 0.05).
# We FAIL TO REJECT H0. There is no evidence the US responds more strongly
# to Ad_Budget than Canada - if anything the point estimate points the
# opposite way, but it's not distinguishable from zero.

# ===========================================================================
# PART C - Chow test: is the ENTIRE Sales relationship different for
# US vs Canada (not just the Ad_Budget slope)?
# ===========================================================================
# Following Cheat Sheet 9's method: build a dummy for the group (here,
# Country, instead of a time break at Obs 50) plus an interaction between
# that dummy and every regressor. Then jointly test whether the dummy and
# all interactions are zero. If we reject, the whole model - intercept and
# every slope - differs between groups, so the two countries should not be
# pooled into a single equation.
data['US'] = np.where(data['Country'] == 'US', 1, 0)          # C0: intercept shift
data['Ad_Budget_US'] = data['US'] * data['Ad_Budget']          # C1: Ad_Budget slope shift
data['Price_US'] = data['US'] * data['Price']                  # C2: Price slope shift
data['Distance_US'] = data['US'] * data['Distance']            # C3: Distance slope shift

model_c = ols(
    'Sales ~ Ad_Budget + Price + Distance + US + Ad_Budget_US + Price_US + Distance_US',
    data
).fit()
print(model_c.summary())

chow_hypothesis = '(US = 0, Ad_Budget_US = 0, Price_US = 0, Distance_US = 0)'
chow_result = model_c.wald_test(chow_hypothesis)
print(chow_result)
# Result: F is large and the p-value is far below 0.05 -> we REJECT the
# null that US and Canada share the same regression equation. The two
# groups are structurally different and should not be pooled.
#
# Reconciling with Part b: the Chow test is a JOINT test across the
# intercept + all three slopes. Part b only tested the Ad_Budget slope in
# isolation and found no difference there. Looking at model_c's individual
# t-tests, the US dummy (intercept shift) is the term driving the Chow
# rejection - Canadians and Americans have a different baseline sales
# level - while none of the individual slope-interaction terms
# (Ad_Budget_US, Price_US, Distance_US) are significant on their own. So
# "structurally different" here means a different starting point (level),
# not different sensitivity to price, distance, or advertising.
