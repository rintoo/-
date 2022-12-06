# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#############################################################################################
####################    Facebook MMM Open Source 'Robyn' Beta - V21.0  ######################
####################                    2021-03-03                     ######################
#############################################################################################

################################################################
#### set locale for non English R
# Sys.setlocale("LC_TIME", "C")

################################################################
#### load libraries
## R version 4.0.3 (2020-10-10) ## Update to R version 4.0.3 to avoid potential errors
## RStudio version 1.2.1335
rm(list=ls()); gc()

## Please make sure to install all libraries before rurnning the scripts
library(data.table) 
library(stringr) 
library(lubridate) 
library(doFuture)
library(doRNG)
library(foreach) 
library(glmnet) 
library(car) 
library(StanHeaders)
library(prophet)
library(rstan)
library(ggplot2)
library(gridExtra)
library(grid)
library(ggpubr)
library(see)
library(PerformanceAnalytics)
library(nloptr)
library(minpack.lm)
library(rPref)
library(reticulate)
library(rstudioapi)
library(parallel)
library(readxl)

## please see https://rstudio.github.io/reticulate/index.html for info on installing reticulate
# conda_create("r-reticulate") # must run this line once
# conda_install("r-reticulate", "nevergrad", pip=TRUE)  #  must install nevergrad in conda before running Robyn
# use_python("/Users/gufengzhou/Library/r-miniconda/envs/r-reticulate/bin/python3.6") # in case nevergrad still can't be imported after installation, please locate your python file and run this line
use_condaenv("r-reticulate") 

################################################################
#### load data & scripts
script_path <- str_sub(rstudioapi::getActiveDocumentContext()$path, start = 1, end = max(unlist(str_locate_all(rstudioapi::getActiveDocumentContext()$path, "/"))))
dt_input <- fread(paste0(script_path,'clean_data.csv'))  # input time series should be daily, weekly or monthly
dt_holidays <- fread(paste0(script_path,'test_holidays_label.csv')) # when using own holidays, please keep the header c("ds", "holiday", "country", "year")

source(paste(script_path, "fb_robyn.func.R", sep=""))
source(paste(script_path, "fb_robyn.optm.R", sep=""))

################################################################
#### set model input variables

set_country <- "CN" # only one country allowed once. Including national holidays for 59 countries, whose list can be found on our githut guide 
set_dateVarName <- c("DATE") # date format must be "2020-01-01"
set_depVarName <- c("TTL_P") # there should be only one dependent variable
set_depVarType <- "revenue" # "revenue" or "conversion" are allowed

activate_prophet <- T # Turn on or off the Prophet feature
# trend在各渠道存在共线性的情况下，会表现为负值，如果共线性严重的情况下，数值较高，会有比较好的拟合效果，数值地，则会纵容共线性的现象
# trend上述情况解释为：ttl_p上升的情况下，并不能给model的预测带来有效提升，相反会加重共线性，提高整体平滑化。trend的负贡献有效削弱了共线性的问题。
# trend在各渠道不存在共线性的情况下，会表现为正值，表现为总体的趋势贡献
set_prophet <- c("trend", "season", "holiday") # "trend","season", "weekday", "holiday" are provided and case-sensitive. Recommended to at least keep Trend & Holidays
set_prophetVarSign <- c("default","default", "default") # c("default", "positive", and "negative"). Recommend as default. Must be same length as set_prophet


activate_baseline <- T
set_baseVarName <- c("TTL_Depth" ) #	"Live_LJQ_Session",	"Live_Viya_Session" typically competitors, price & promotion, temperature,  unemployment rate etc
set_baseVarSign <- c("positive") #,"positive", "positive" c("default", "positive", and "negative"), control the signs of coefficients for baseline variables

set_mediaVarName <- c("OTV","Display_Social","Display_Other","Live_Others_SPD","Wechat_Social_SPD","Weibo_Social_SPD","ZTC_SPD","Banner_SPD","Starstore_Brand","CJTJ_SPD","Texiu_SPD","TikTok_Social_SPD","Red_Social_SPD","Live_LJQ_SPD","Live_Viya_SPD","TikTok_RTB_Imp") 
set_mediaVarSign <- c("positive", "positive", "positive","positive", "positive", "positive","positive", "positive", "positive", "positive","positive", "positive", "positive", "positive", "positive", "positive") # c("default", "positive", and "negative"), control the signs of coefficients for media variables
set_mediaSpendName <- c("OTV","Display_Social","Display_Other","Live_Others_SPD","Wechat_Social_SPD","Weibo_Social_SPD","ZTC_SPD","Banner_SPD","Starstore_Brand","CJTJ_SPD","Texiu_SPD","TikTok_Social_SPD","Red_Social_SPD","Live_LJQ_SPD","Live_Viya_SPD","TikTok_RTB_Spending") # spends must have same order and same length as set_mediaVarName

set_factorVarName <- c() # please specify which variable above should be factor, otherwise leave empty c()

################################################################
#### set global model parameters

## set cores for parallel computing######################
registerDoSEQ(); detectCores()
set_cores <- 14 # I am using 6 cores from 8 on my local machine. Use detectCores() to find out cores

## set rolling window start (only works for whole dataset for now)
# set_trainStartDate <- unlist(dt_input[, lapply(.SD, function(x) as.character(min(x))), .SDcols= set_dateVarName])
set_trainStartDate <- unlist(dt_input[, lapply(.SD, function(x) as.character(min(x))), .SDcols= set_dateVarName])

## set model core features
adstock <- "geometric" # geometric or weibull. weibull is more flexible, yet has one more parameter and thus takes longer
set_iter <- 500  # number of allowed iterations per trial. 500 is recommended

set_hyperOptimAlgo <- "DiscreteOnePlusOne" # selected algorithm for Nevergrad, the gradient-free optimisation library https://facebookresearch.github.io/nevergrad/index.html
set_trial <- 40 # number of allowed iterations per trial. 40 is recommended without calibration, 100 with calibration.
## Time estimation: with geometric adstock, 500 iterations * 40 trials and 6 cores, it takes less than 1 hour. Weibull takes at least twice as much time.

## helper plots: set plot to TRUE for transformation examples
f.plotAdstockCurves(F) # adstock transformation example plot, helping you understand geometric/theta and weibull/shape/scale transformation
f.plotResponseCurves(F) # s-curve transformation example plot, helping you understand hill/alpha/gamma transformation

################################################################
#### tune channel hyperparameters bounds

#### Guidance to set hypereparameter bounds #### 

## 1. get correct hyperparameter names: 
local_name <- f.getHyperNames(); local_name # names in set_hyperBoundLocal must equal names in local_name, case sensitive

## 2. get guidance for setting hyperparameter bounds:
# For geometric adstock, use theta, alpha & gamma. For weibull adstock, use shape, scale, alpha, gamma
# theta: In geometric adstock, theta is decay rate. guideline for usual media genre: TV c(0.3, 0.8), OOH/Print/Radio c(0.1, 0.4), digital c(0, 0.3)
# shape: In weibull adstock, shape controls the decay shape. Recommended c(0.0001, 2). The larger, the more S-shape. The smaller, the more L-shape
# scale: In weibull adstock, scale controls the decay inflexion point. Very conservative recommended bounce c(0, 0.1), becausee scale can increase adstocking half-life greaetly
# alpha: In s-curve transformation with hill function, alpha controls the shape between exponential and s-shape. Recommended c(0.5, 3). The larger the alpha, the more S-shape. The smaller, the more C-shape
# gamma: In s-curve transformation with hill function, gamma controls the inflexion point. Recommended bounce c(0.3, 1). The larger the gamma, the later the inflection point in the response curve 

## 3. set each hyperparameter bounds. They either contains two values e.g. c(0, 0.5), or only one value (in which case you've "fixed" that hyperparameter)
# theta越大滑动延迟效应越严重。eg theta = 0.75 的广告素材意味着第 1 期中 75% 的展示被带到第 2 期
# alphas和gammas被认为是广告的收益递减参数
set_hyperBoundLocal <- list(
  Banner_SPD_alphas= c(0.5, 3),        Banner_SPD_gammas= c(0.3, 1),     Banner_SPD_thetas= c(0, 0.3),        
  Starstore_Brand_alphas= c(0.5, 3),  Starstore_Brand_gammas= c(0.3, 1),     Starstore_Brand_thetas= c(0.1, 0.5),    
  CJTJ_SPD_alphas= c(0.5, 3),            CJTJ_SPD_gammas= c(0.3, 1),         CJTJ_SPD_thetas= c(0, 0.1),   
  Live_LJQ_SPD_alphas= c(0.5, 3),        Live_LJQ_SPD_gammas= c(0.3, 1),      Live_LJQ_SPD_thetas= c(0.1, 0.7),      
  Live_Viya_SPD_alphas= c(0.5, 3),       Live_Viya_SPD_gammas= c(0.3, 1),     Live_Viya_SPD_thetas= c(0, 0.5),  
  Red_Social_SPD_alphas= c(0.5, 3),     Red_Social_SPD_gammas= c(0.3, 1),  Red_Social_SPD_thetas= c(0, 0.3),
  #Starstore_SPD_alphas= c(0.5, 3),       Starstore_SPD_gammas= c(0.3, 1),    Starstore_SPD_thetas= c(0.1, 0.3),     
  Texiu_SPD_alphas= c(0.5, 3),        Texiu_SPD_gammas= c(0.3, 1),         Texiu_SPD_thetas= c(0, 0.3),
  TikTok_RTB_Imp_alphas= c(0.5, 3),      TikTok_RTB_Imp_gammas= c(0.3, 1),    TikTok_RTB_Imp_thetas= c(0, 0.1),    
  TikTok_Social_SPD_alphas= c(0.5, 3),   TikTok_Social_SPD_gammas= c(0.3, 1),TikTok_Social_SPD_thetas= c(0, 0.3),
  ZTC_SPD_alphas= c(0.5, 3),            ZTC_SPD_gammas= c(0.3, 1),          ZTC_SPD_thetas= c(0, 0.1)  ,
  Live_Others_SPD_alphas= c(0.5, 3),       Live_Others_SPD_gammas= c(0.3, 1),    Live_Others_SPD_thetas= c(0, 0.5),
  Wechat_Social_SPD_alphas= c(0.5, 3),     Wechat_Social_SPD_gammas= c(0.3, 1),  Wechat_Social_SPD_thetas= c(0, 0.3),
  Weibo_Social_SPD_alphas= c(0.5, 3),     Weibo_Social_SPD_gammas= c(0.3, 1),  Weibo_Social_SPD_thetas= c(0, 0.3),
  OTV_alphas= c(0.5, 3), OTV_gammas= c(0.3, 1), OTV_thetas = c(0, 0.5),
  Display_Other_alphas= c(0.5, 3), Display_Other_gammas= c(0.3, 1), Display_Other_thetas = c(0, 0.5),
  Display_Social_alphas= c(0.5, 3), Display_Social_gammas= c(0.3, 1), Display_Social_thetas = c(0.1, 0.3)
  
)
# 
# temp = c()
# for (i in 1:length(local_name)){
#   if(i%%3==0){
#     temp[i] = paste0(local_name[i],"= c(0.5, 3),")
#   }
#   if(i%%3==1){
#     temp[i] = paste0(local_name[i],"= c(0.3, 1),")
#   }
#   if(i%%3==2){
#     temp[i] = paste0(local_name[i],"= c(0.3, 0.8),")
#   }
# }

################################################################
#### define ground truth (e.g. Geo test, FB Lift test, MTA etc.)

activate_calibration <- F # Switch to TRUE to calibrate model.
# set_lift <- data.table(channel = c("facebook_I",  "tv_S", "facebook_I"),
#                        liftStartDate = as.Date(c("2018-05-01", "2017-11-27", "2018-07-01")),
#                        liftEndDate = as.Date(c("2018-06-10", "2017-12-03", "2018-07-20")),
#                        liftAbs = c(400000, 300000, 200000))

################################################################
#### Prepare input data

dt_mod <- f.inputWrangling() 

################################################################
#### Run models

model_output_collect <- f.robyn(set_hyperBoundLocal
                                ,optimizer_name = set_hyperOptimAlgo
                                ,set_trial = set_trial
                                ,set_cores = set_cores
                                ,plot_folder = getwd()) # please set your folder path to save plots. It ends without "/".

## reload old models from csv

# dt_hyppar_fixed <- fread("/Users/gufengzhou/Documents/GitHub/plots/2021-04-07 09.12/pareto_hyperparameters.csv") # load hyperparameter csv. Provide your own path.
# model_output_collect <- f.robyn.fixed(plot_folder = "~/Documents/GitHub/plots", dt_hyppar_fixed = dt_hyppar_fixed[solID == "2_16_5"]) # solID must be included in the csv

################################################################
#### Budget Allocator - Beta

## Budget allocator result requires further validation. Please use this result with caution.
## Please don't interpret budget allocation result if there's no satisfying MMM result

model_output_collect$allSolutions
optim_result <- f.budgetAllocator(modID = "19_35_10" # input one of the model IDs in model_output_collect$allSolutions to get optimisation result
                                  ,scenario = "max_historical_response" # c(max_historical_response, max_response_expected_spend)
                                  #,expected_spend = 100000 # specify future spend volume. only applies when scenario = "max_response_expected_spend"
                                  #,expected_spend_days = 90 # specify period for the future spend volumne in days. only applies when scenario = "max_response_expected_spend"
                                  ,channel_constr_low = c(0.7, 0.75, 0.60, 0.8, 0.65,0.7, 0.75, 0.60, 0.8, 0.65,0.8) # must be between 0.01-1 and has same length and order as set_mediaVarName
                                  ,channel_constr_up = c(1.2, 1.5, 1.5, 2, 1.5,1.2, 1.5, 1.5, 2, 1.5,1.3) # not recommended to 'exaggerate' upper bounds. 1.5 means channel budget can increase to 150% of current level
)
