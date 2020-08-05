The M5 Forecasting Accuracy Competition
================================================

Nousot competed in a kaggle competition,
[M5 Forecasting - Accuracy](https://www.kaggle.com/c/m5-forecasting-accuracy/data),
that challenges competitors to predict item sales at stores in various locations
for 28-days time periods. The team took 16th place, and we wrote a
[blog article](https://nousot.com/blog/how-we-won-gold/) describing the experience.
This documentation shows how
to reproduce the contest submission. Things got a little hairy at the end, so
please add an issue if something needs attention.


See [M5 Documentation here](https://mk0mcompetitiont8ake.kinstacdn.com/wp-content/uploads/2020/02/M5-Competitors-Guide_Final-1.pdf).

# Reproducing the Final Submission.
  - Run `feature-eng.py` after setting the competition data set directory
    and output directories
  - Run `item-modeling-ss.R` after setting the input directory to the location
    of the csv.gz file created in the first step, and noting the output location
    of the image file at the end of the script and the submission file
    `shades-submission-v1.csv`.
  - Run steps in section below, Reproducing the Dark Shades solution, to create
    the file `shades-submission-v1.csv`
  - Run the R script `submission-work.R` up to the line that writes the file
    `kalman-lgb-loess-choose-final.csv`. This was the final submission. The 
     reading in of the image file was a convenience as time was running out.

# Reproducing the Dark Shades solution:
All files in the list below are in the `solutions` directory

 - Create grid pickles parts 1-3 by running `m5-simple-fe.py`,
      changing data paths if necessary. The outputs are:
   - `grid_part_1.pkl` is long form of item sales, tossing out leading zeros 
   - `grid_part_2.pkl` is 1 with price featured merged in, including
      "price momentum" at the yearly and monthly level
   - `grid_part_3.pkl` is 1 with cal featured merged in and integer cal features
      included like day of month, etc
 - Create lags features by running `m5-lags-features.py`, noting the data
       input paths and reliance on the grid pickles. The output is:
   -  `lags_df_28.pkl`, lags and rolling means, sds, for different periods,
          using only store `CA_1` to save computation
 - Create encodings features by running `m5-custom-features.py`, producing the
      output:
   - `mean_encoding_df.pkl`, numerical encodings of the state, category, dept, etc
 - Run the Dark Shades flow, noting file paths of all outputs discussed above,
       by running the file `shades-of-dark-magic.py`. This will produce the
       submission file `shades-submission_v1.csv`
