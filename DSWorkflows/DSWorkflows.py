from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split

class Workflow:
    """
    This class serves to encapsulate the whole pipeline for a ML project and simplify the process of data analysis, preprocessing, crossvalidation and evaluation.

    ## Parameters
        dataframe: Model data in Pandas format.
        seed: Numeric value for reproducibility. (Default: 100)
    """
    def __init__(self, dataframe: pd.DataFrame, seed: int = 100):
        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError('Input must be a Pandas DataFrame')
        self.dataframe = dataframe
        self.seed = seed
        self.best_pipeline = None
        self.best_score = None

    def EDA(self, scaling_factor: float = 1.0, graph_limit: int = 10, high_corr: float = 0.75):
        """
        Performs and exploratory data analysis. Returns the results directly into the interactive console.
        ## Parameters
            scaling_factor: Float value for adjusting graph size. (Default: 1.0)
            graph_limit: Maximum number of graphs to display for each group (scatter, histograms and bar). (Default: 10, Minimum: 2)
            high_corr: Defines the correlation threshold to print out the scatter graphs. (Default: 0.75)
        """
        if graph_limit < 2:
            graph_limit = 2

        # Dataset Shape
        print('Data Shape:')
        display(self.dataframe.shape)
        
        # General information for the dataset
        print('General information:')
        self.dataframe.info()

        # Create plots and figures depending on the type of variable
        numericals = self.dataframe.select_dtypes(exclude='object')
        categoricals = self.dataframe.select_dtypes(include='object')

        if not numericals.empty:
            # Numeric variable analysis
            print('\nNumerical variable analysis:')
            display(self.dataframe.describe())
        if not categoricals.empty:
            # Numeric variable analysis
            print('\nCategorical variable analysis:')
            display(self.dataframe.describe(include='object'))

        # Get duplicated values
        print('\nDuplicated Values:')
        display(self.dataframe.duplicated().sum())

        # Get null values
        print('\nNull Values:')
        display(self.dataframe.isnull().sum())

        # Define color scheme
        colors = plt.rcParams["axes.prop_cycle"]()
        
        if not numericals.empty:
            # Plot correlation matrix
            corr_data = numericals.corr()
            fig, ax = plt.subplots(figsize=(len(numericals.columns)*scaling_factor, len(numericals.columns)*scaling_factor))
            im = ax.imshow(corr_data, vmin=-1, vmax=1)
            # Show all ticks and label them with the respective list entries
            ax.set_xticks(np.arange(len(corr_data.columns)), labels=corr_data.columns)
            ax.set_yticks(np.arange(len(corr_data.columns)), labels=corr_data.columns)
            # Rotate the tick labels and set their alignment.
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor', fontsize=14)
            plt.setp(ax.get_yticklabels(), fontsize=14)
            high_correlation = []
            # Loop over data dimensions and create text annotations.
            for i in range(len(corr_data.columns)):
                for j in range(len(corr_data.columns)):
                    corr = round(corr_data.iloc[i, j], 2)
                    ax.text(j, i, corr, ha='center', va='center', color='w', fontsize=14)
                    if abs(corr) >= high_corr and i != j:
                        if [corr_data.columns[i], corr_data.columns[j]] not in high_correlation:
                            if [corr_data.columns[j], corr_data.columns[i]] not in high_correlation:
                                high_correlation.append([corr_data.columns[i], corr_data.columns[j]])
            ax.set_title('Correlation Matrix', fontsize=20)
            fig.tight_layout()
            # Create colorbar
            ax.figure.colorbar(im, ax=ax, orientation='vertical', label='Correlation', shrink=0.7, pad=0.05)

            if high_correlation:
                # Plot scatter of highly correlated variables
                if len(high_correlation) > graph_limit:
                    row_num = graph_limit
                else:
                    row_num = len(high_correlation)
                fig2, ax2 = plt.subplots(nrows=row_num, ncols=1, figsize=(6*scaling_factor, 4*row_num*scaling_factor), constrained_layout=True)
                if len(high_correlation) > 1:
                    for i, corr_pair in enumerate(high_correlation):
                        if i < row_num:
                            ax2[i].scatter(numericals[corr_pair[0]], numericals[corr_pair[1]], c=next(colors)["color"])
                            ax2[i].set_xlabel(corr_pair[0])
                            ax2[i].set_ylabel(corr_pair[1])
                        else:
                            break
                else:
                    ax2.scatter(numericals[high_correlation[0][0]], numericals[high_correlation[0][1]], c=next(colors)["color"])
                    ax2.set_xlabel(high_correlation[0][0])
                    ax2.set_ylabel(high_correlation[0][1])

                fig2.suptitle('Highly Correlated Variables')

            # Plot distribution histograms
            if len(numericals.columns) > graph_limit:
                row_num = graph_limit
            else:
                row_num = len(numericals.columns)
            fig3, ax3 = plt.subplots(nrows=row_num, ncols=1, figsize=(8*scaling_factor, 3*row_num*scaling_factor), constrained_layout=True)
            
            for i, column_name in enumerate(numericals):
                if i < row_num:
                    ax3[i].hist(numericals[column_name], bins=30, color=next(colors)["color"]) # Draw histogram
                    m = numericals[column_name].mean() # Calculate mean
                    ax3[i].axvline(x=m, color='red', linestyle='--', linewidth=2, label='Avg') # Draw mean line
                    ax3[i].text(x=m, y=ax3[i].get_ylim()[1]*0.95, s='Mean: '+str(round(m, 2)), color='red', ha='left') # Place text with mean value close to mean line
                    ax3[i].set_xlabel(column_name)
                    ax3[i].set_ylabel('Frequency')
                else:
                    break

            fig3.suptitle('Numeric Variable Distributions')

        if not categoricals.empty:
            if len(categoricals.columns) > graph_limit:
                row_num = graph_limit
            else:
                row_num = len(categoricals.columns)
            fig4, ax4 = plt.subplots(nrows=row_num, ncols=1, figsize=(6*scaling_factor, 4*row_num*scaling_factor), constrained_layout=True)
            for i, ax in enumerate(ax4):
                # Plot barchart from Pandas
                if i < row_num:
                    categoricals[categoricals.columns[i]].value_counts().iloc[:row_num].plot.barh(ax=ax, xlabel='Count', ylabel=categoricals.columns[i], rot=0, color=next(colors)["color"])
                else:
                    break

            fig4.suptitle('Categorical Variable Distributions')

        plt.show()
        
    def get_X_and_y(self, target_name: str, remove_duplicates=False):
        """
        Return model features (X) and target variable (y).

        ## Parameters
            target_name: Name of y in the dataframe.
            remove_duplicates: Return values with duplicates removed. (Default: False)
        """
        if remove_duplicates:
            df = self.dataframe.drop_duplicates() # Remove duplicates
        else:
            df = self.dataframe
        X = df.drop(columns=[self.target_name])
        y = df[self.target_name]
        return X, y
    
    def undersample_data(self, X, y):
        """
        Returns an undersampled version of the data using ImbLearn.
        """
        rus = RandomUnderSampler(random_state=self.seed)
        return rus.fit_resample(X, y)

    def oversample_data(self, X, y):
        """
        Returns an oversampled version of the data using ImbLearn.
        """
        rus = RandomOverSampler(random_state=self.seed)
        return rus.fit_resample(X, y)

    def evaluate(self, pipelines: list, X, y, test_type: str, n_splits: int = 5, multiclass_avg: str = 'binary', scaling_factor: float = 1.0):
        """
        This function receives the pipelines to be crossvalidated, takes the one with the highest score and applies the appropiate evaluation method

        ## Parameters
            pipelines: Model pipelines to be evaluated.
            X: Model features.
            y: Target variable.
            test_type: Evaluation method for crossvalidation.
            n_splits: Number of splits for crossvalidation. (Default: 5)
            multiclass_avg: Calculation method for classification metrics. Set to 'micro', 'macro', 'samples', 'weighted' or None in multilabel and 'binary' for binary classification. (Default: binary)
            scaling_factor: Float value for adjusting graph size. (Default: 1.0)
        """
        classification = [
            'accuracy',
            'balanced_accuracy',
            'top_k_accuracy',
            'average_precision',
            'neg_brier_score',
            'f1',
            'f1_micro',
            'f1_macro',
            'f1_weighted',
            'f1_samples',
            'neg_log_loss',
            'precision',
            'recall',
            'jaccard',
            'roc_auc',
            'roc_auc_ovr',
            'roc_auc_ovo',
            'roc_auc_ovr_weighted',
            'roc_auc_ovo_weighted'
        ]

        regression = [
            'explained_variance',
            'max_error',
            'neg_mean_absolute_error',
            'neg_mean_squared_error',
            'neg_root_mean_squared_error',
            'neg_mean_squared_log_error',
            'neg_median_absolute_error',
            'r2',
            'neg_mean_poisson_deviance',
            'neg_mean_gamma_deviance',
            'neg_mean_absolute_percentage_error',
            'd2_absolute_error_score',
            'd2_pinball_score',
            'd2_tweedie_score',
        ]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = self.seed)
        highscore = [None, 0.00, 0]

        # Calculate mean and standard deviation for all pipelines and select the best one
        for i, pipeline in enumerate(pipelines):
            scores = cross_val_score(pipeline, X_train, y_train, scoring=test_type, cv=n_splits, n_jobs=-1)
            mean = np.mean(scores)
            print(f'Pipeline {i}:', scores, f'Mean score: {mean} +/- {np.std(scores)} stdev')

            if mean > highscore[1]:
                highscore[0] = pipeline
                highscore[1] = mean
                highscore[2] = i
        
        if not self.best_pipeline or not self.best_score:
            self.best_pipeline = highscore[0]
            self.best_score = highscore[1]

        if highscore[1] > self.best_score:
            self.best_pipeline = highscore[0]
            self.best_score = highscore[1]
            
        highscore[0].fit(X_train, y_train)
        y_pred = highscore[0].predict(X_test)

        # Determine what type of problem is being solved and print appropiate metrics
        if test_type in regression:
            print(f'\nPrinting results for pipeline {highscore[2]}:')
            print('MSE:', mean_squared_error(y_test, y_pred))
            print('R-Squared:', r2_score(y_test, y_pred))
            print('MAPE:', mean_absolute_percentage_error(y_test, y_pred))
            print('MAE:', mean_absolute_error(y_test, y_pred))
        elif test_type in classification:
            labels = len(pd.Series(y).unique())
            print(f'\nPrinting results for pipeline {highscore[2]}:')
            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            fig, ax = plt.subplots(figsize=(labels*scaling_factor,labels*scaling_factor))
            disp.plot(ax=ax)

            print('Accuracy:', accuracy_score(y_test, y_pred))
            print('Precission:', precision_score(y_test, y_pred, average=multiclass_avg)) 
            print('Recall:', recall_score(y_test, y_pred, average=multiclass_avg))
            print('F1 score:', f1_score(y_test, y_pred, average=multiclass_avg))
        else:
            print('Couldn\'t identify a proper evaluation method')