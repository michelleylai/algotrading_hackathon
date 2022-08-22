#! /bin/sh
# Copyright Joses Ho and Michelle Lai 2017
##########################################


 #      # #####  #####    ##   #####  # ######  ####
 #      # #    # #    #  #  #  #    # # #      #
 #      # #####  #    # #    # #    # # #####   ####
 #      # #    # #####  ###### #####  # #           #
 #      # #    # #   #  #    # #   #  # #      #    #
 ###### # #####  #    # #    # #    # # ######  ####

from quantopian.algorithm import attach_pipeline, pipeline_output, order_optimal_portfolio

from quantopian.pipeline import Pipeline

from quantopian.pipeline.factors import CustomFactor, SimpleMovingAverage, AverageDollarVolume, RollingLinearRegressionOfReturns

from quantopian.pipeline.filters.morningstar import IsPrimaryShare

from quantopian.pipeline.classifiers.morningstar import Sector

from quantopian.pipeline.data import morningstar
from quantopian.pipeline.data import Fundamentals
from quantopian.pipeline.data import morningstar
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.data.psychsignal import stocktwits

import numpy as np
import pandas as pd
from scipy import stats

from quantopian.pipeline.filters import Q1500US
import quantopian.optimize as opt

WINLENGTH=30
BASE_UNIVERSE = Q1500US()

  ####   ####  #    #  ####  ##### #####    ##   # #    # #####  ####
 #    # #    # ##   # #        #   #    #  #  #  # ##   #   #   #
 #      #    # # #  #  ####    #   #    # #    # # # #  #   #    ####
 #      #    # #  # #      #   #   #####  ###### # #  # #   #        #
 #    # #    # #   ## #    #   #   #   #  #    # # #   ##   #   #    #
  ####   ####  #    #  ####    #   #    # #    # # #    #   #    ####

# Constraint Parameters
MAX_GROSS_EXPOSURE = 1.0
# NUM_LONG_POSITIONS = 300
# NUM_SHORT_POSITIONS = 300

# Risk Exposures
MAX_SECTOR_EXPOSURE = 0.05 # hackathon requirement.
MAX_BETA_EXPOSURE = 0.10 # unchanged, to reduce?

 #####   ####   ####  # ##### #  ####  #    #
 #    # #    # #      #   #   # #    # ##   #
 #    # #    #  ####  #   #   # #    # # #  #
 #####  #    #      # #   #   # #    # #  # #
 #      #    # #    # #   #   # #    # #   ##
 #       ####   ####  #   #   #  ####  #    #
  ####  # ###### ######
 #      #     #  #
  ####  #    #   #####
      # #   #    #
 #    # #  #     #
  ####  # ###### ######

# Here we define the maximum position size that can be held for any
# given stock. If you have a different idea of what these maximum
# sizes should be, feel free to change them. Keep in mind that the
# optimizer needs some leeway in order to operate. Namely, if your
# maximum is too small, the optimizer may be overly-constrained. [AXN]
MAX_SHORT_POSITION_SIZE = 0.10
MAX_LONG_POSITION_SIZE = 0.10

  #####
 #     # #    #  ####  #####  ####  #    #
 #       #    # #        #   #    # ##  ##
 #       #    #  ####    #   #    # # ## #
 #       #    #      #   #   #    # #    #
 #     # #    # #    #   #   #    # #    #
  #####   ####   ####    #    ####  #    #
 #######
 #         ##    ####  #####  ####  #####   ####
 #        #  #  #    #   #   #    # #    # #
 #####   #    # #        #   #    # #    #  ####
 #       ###### #        #   #    # #####       #
 #       #    # #    #   #   #    # #   #  #    #
 #       #    #  ####    #    ####  #    #  ####


 ######  ### ######  ####### #       ### #     # #######
 #     #  #  #     # #       #        #  ##    # #
 #     #  #  #     # #       #        #  # #   # #
 ######   #  ######  #####   #        #  #  #  # #####
 #        #  #       #       #        #  #   # # #
 #        #  #       #       #        #  #    ## #
 #       ### #       ####### ####### ### #     # #######

def my_pipe():

    # Construct Factors.
    sma_kwargs=dict(window_length=WINLENGTH,mask=BASE_UNIVERSE)

    latest_pb=morningstar.valuation_ratios.pb_ratio.latest
    latest_payout=morningstar.valuation_ratios.payout_ratio.latest

    sma_pb = SimpleMovingAverage(inputs=[morningstar.valuation_ratios.pb_ratio],**sma_kwargs)
    sma_payout = SimpleMovingAverage(inputs=[morningstar.valuation_ratios.payout_ratio],**sma_kwargs)

    # filters
    mkt_cap_lower = morningstar.valuation.market_cap.latest >= 250000000 # $250MM
    mkt_cap_upper = morningstar.valuation.market_cap.latest <= 5000000000 # $5 Billion
    price_filter = USEquityPricing.close.latest >= 5

    pb_filter=(sma_pb<=15)
    payout_filter=(sma_payout<=0.5)

    # with market cap filter
    # allfilters=mkt_cap_lower & mkt_cap_upper & price_filter & pb_filter & payout_filter
    allfilters=mkt_cap_lower & price_filter & pb_filter & payout_filter

    universe = BASE_UNIVERSE & allfilters

    # using 30day average
    alpha = ((1/sma_pb.zscore()*0.6) +
             sma_payout.zscore()*0.4) / 2

    # # using 1day average
    # alpha = ((1/latest_pb.zscore()*0.6) +
    #          latest_payout.zscore()*0.4) / 2

    # Build Filters representing the top and bottom 150 stocks by our combined ranking system.
    # We'll use these as our tradeable universe each day.
    longs = alpha.percentile_between(70, 100)
    shorts = alpha.percentile_between(0, 30)

    # The final output of our pipeline should only include
    # the top/bottom 300 stocks by our criteria
    long_short_screen = (longs | shorts)

    # Define any risk factors that we will want to neutralize
    # We are chiefly interested in market beta as a risk factor so we define it using
    # Bloomberg's beta calculation
    # Ref: https://www.lib.uwo.ca/business/betasbydatabasebloombergdefinitionofbeta.html
    beta = 0.66*RollingLinearRegressionOfReturns(
                    target=sid(8554),
                    returns_length=5,
                    regression_length=260,
                    mask=long_short_screen
                    ).beta + 0.33*1.0

    pipeout=Pipeline( columns={'alpha' : alpha,
                               'beta' : beta,
                               'sector' : Sector()},
                     screen=universe # screen pipeline output on filtered BASE_UNIVERSE
                   )
    return pipeout

 # #    # # ##### #   ##   #      # ###### ######
 # ##   # #   #   #  #  #  #      #     #  #
 # # #  # #   #   # #    # #      #    #   #####
 # #  # # #   #   # ###### #      #   #    #
 # #   ## #   #   # #    # #      #  #     #
 # #    # #   #   # #    # ###### # ###### ######

   ####   ####  #    # ###### #####  #    # #      ######
  #      #    # #    # #      #    # #    # #      #
   ####  #      ###### #####  #    # #    # #      #####
       # #      #    # #      #    # #    # #      #
  #    # #    # #    # #      #    # #    # #      #
   ####   ####  #    # ###### #####   ####  ###### ######
     #####  ###### #####    ##   #        ##   #    #  ####  ######
     #    # #      #    #  #  #  #       #  #  ##   # #    # #
     #    # #####  #####  #    # #      #    # # #  # #      #####
     #####  #      #    # ###### #      ###### #  # # #      #
     #   #  #      #    # #    # #      #    # #   ## #    # #
     #    # ###### #####  #    # ###### #    # #    #  ####  ######

def initialize(context):
    """
    Called at the start of each day.
    """
    # Here we set our slippage and commisions. Set slippage
    # and commission to zero to evaulate the signal-generating
    # ability of the algorithm independent of these additional
    # costs.
    set_commission(commission.PerShare(cost=0.0, min_trade_cost=0))
    set_slippage(slippage.VolumeShareSlippage(volume_limit=1, price_impact=0))
    context.spy = sid(8554)

    attach_pipeline(my_pipe(), 'two_factor_pipeline')

    # Schedule my rebalance function
    schedule_function(func=rebalance,
                      date_rule=date_rules.month_start(),
                      time_rule=time_rules.market_open(hours=0,minutes=30),
                      half_days=True)
    # record my portfolio variables at the end of day
    schedule_function(func=recording_statements,
                      date_rule=date_rules.every_day(),
                      time_rule=time_rules.market_close(),
                      half_days=True)

 #####  ###### ######  ####  #####  ######
 #    # #      #      #    # #    # #
 #####  #####  #####  #    # #    # #####
 #    # #      #      #    # #####  #
 #    # #      #      #    # #   #  #
 #####  ###### #       ####  #    # ######
 ##### #####    ##   #####  # #    #  ####
   #   #    #  #  #  #    # # ##   # #    #
   #   #    # #    # #    # # # #  # #
   #   #####  ###### #    # # #  # # #  ###
   #   #   #  #    # #    # # #   ## #    #
   #   #    # #    # #####  # #    #  ####
  ####  #####   ##   #####  #####
 #        #    #  #  #    #   #
  ####    #   #    # #    #   #
      #   #   ###### #####    #
 #    #   #   #    # #   #    #
  ####    #   #    # #    #   #

def before_trading_start(context, data):
    """
    Called every day before trading starts.
    """
    # Call pipeline_output to get the output
    # Note: this is a dataframe where the index is the SIDs for all
    # securities to pass my screen and the columns are the factors
    # added to the pipeline object above
    context.pipeline_data = pipeline_output('two_factor_pipeline')


def recording_statements(context, data):
    """
    Set which variables are plotted.
    """
    # Plot the number of positions over time.
    record(num_positions=len(context.portfolio.positions))

 #####  ###### #####    ##   #        ##   #    #  ####  ######
 #    # #      #    #  #  #  #       #  #  ##   # #    # #
 #    # #####  #####  #    # #      #    # # #  # #      #####
 #####  #      #    # ###### #      ###### #  # # #      #
 #   #  #      #    # #    # #      #    # #   ## #    # #
 #    # ###### #####  #    # ###### #    # #    #  ####  ######

# Called at the start of every month in order to rebalance
# the longs and shorts lists
def rebalance(context, data):
    ### Optimize API
    pipeline_data = context.pipeline_data

    ### Extract from pipeline any specific risk factors you want
    # to neutralize that you have already calculated
    risk_factor_exposures = pd.DataFrame({
            'beta':pipeline_data.beta.fillna(1.0)
        })
    # # We fill in any missing factor values with a market beta of 1.0.
    # # We do this rather than simply dropping the values because we have
    # # want to err on the side of caution. We don't want to exclude
    # # a security just because it's missing a calculated market beta,
    # # so we assume any missing values have full exposure to the market.


    ### Here we define our objective for the Optimize API. We have
    # selected MaximizeAlpha because we believe our combined factor
    # ranking to be proportional to expected returns. This routine
    # will optimize the expected return of our algorithm, going
    # long on the highest expected return and short on the lowest.
    objective = opt.MaximizeAlpha(pipeline_data.alpha)

    ### Define the list of constraints
    constraints = []
    # Constrain our maximum gross leverage
    constraints.append(opt.MaxGrossExposure(MAX_GROSS_EXPOSURE))
    # Require our algorithm to remain dollar neutral.
    constraints.append(opt.DollarNeutral(.05))
    # Add a sector neutrality constraint using the sector
    # classifier that we included in pipeline
    constraints.append(
        opt.NetGroupExposure.with_equal_bounds(
            labels=pipeline_data.sector,
            min=-MAX_SECTOR_EXPOSURE,
            max=MAX_SECTOR_EXPOSURE,
        ))
    # Take the risk factors that you extracted above and
    # list your desired max/min exposures to them -
    # Here we selection +/- 0.01 to remain near 0.
    neutralize_risk_factors = opt.FactorExposure(
        loadings=risk_factor_exposures,
        min_exposures={'beta':-MAX_BETA_EXPOSURE},
        max_exposures={'beta':MAX_BETA_EXPOSURE}
    )
    constraints.append(neutralize_risk_factors)

    # With this constraint we enforce that no position can make up
    # greater than MAX_SHORT_POSITION_SIZE on the short side and
    # no greater than MAX_LONG_POSITION_SIZE on the long side. This
    # ensures that we do not overly concentrate our portfolio in
    # one security or a small subset of securities.
    constraints.append(
        opt.PositionConcentration.with_equal_bounds(
            min=-MAX_SHORT_POSITION_SIZE,
            max=MAX_LONG_POSITION_SIZE
        ))

    # Put together all the pieces we defined above by passing
    # them into the order_optimal_portfolio function. This handles
    # all of our ordering logic, assigning appropriate weights
    # to the securities in our universe to maximize our alpha with
    # respect to the given constraints.
    order_optimal_portfolio(
        objective=objective,
        constraints=constraints,
    )
