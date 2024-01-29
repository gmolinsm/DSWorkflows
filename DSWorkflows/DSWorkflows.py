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
        target_name: Name for the target variable.
    """
    def __init__(self, dataframe: pd.DataFrame, target_name: str, seed = 100):
        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError('Input must be a Pandas DataFrame')
        self.dataframe = dataframe
        self.target_name = target_name
        self.seed = seed

    def EDA(self):
        """
        Performs and exploratory data analysis. Returns the results directly into the interactive console.
        """
        # Dataset Shape
        print('Data Shape:')
        display(self.dataframe.shape)
        
        # General information for the dataset
        print('General information:')
        self.dataframe.info()

        # Numeric variable analysis
        print('\nNumeric variable analysis:')
        display(self.dataframe.describe())

        # Get duplicated values
        print('\nDuplicated Values:')
        display(self.dataframe.duplicated().sum())

        # Get null values
        print('\nNull Values:')
        display(self.dataframe.isnull().sum())

        # Create plots depending on the type of variable
        numericals = self.dataframe.select_dtypes(exclude='object')
        categoricals = self.dataframe.select_dtypes(include='object')

        if not numericals.empty:
            n_graphs = len(numericals.columns)
            size = n_graphs*2
            # Plot correlation matrix
            corr_data = numericals.corr()
            fig, ax = plt.subplots(figsize=(size, size))
            im = ax.imshow(corr_data, vmin=-1, vmax=1)
            # Show all ticks and label them with the respective list entries
            ax.set_xticks(np.arange(len(corr_data.columns)), labels=corr_data.columns)
            ax.set_yticks(np.arange(len(corr_data.columns)), labels=corr_data.columns)
            # Rotate the tick labels and set their alignment.
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right',
                    rotation_mode='anchor')
            high_correlation = []
            # Loop over data dimensions and create text annotations.
            for i in range(len(corr_data.columns)):
                for j in range(len(corr_data.columns)):
                    corr = round(corr_data.iloc[i, j], 2)
                    ax.text(j, i, corr, ha='center', va='center', color='w', fontsize=14)
                    if abs(corr) >= 0.75 and i != j:
                        if [corr_data.columns[i], corr_data.columns[j]] not in high_correlation:
                            if [corr_data.columns[j], corr_data.columns[i]] not in high_correlation:
                                high_correlation.append([corr_data.columns[i], corr_data.columns[j]])
            ax.set_title('Correlation Matrix', fontsize=20)
            fig.tight_layout()
            # Create colorbar
            ax.figure.colorbar(im, ax=ax, location='bottom', label='Correlation', shrink=0.7)

            if high_correlation:
                # Plot scatter of highly correlated variables
                fig2, ax2 = plt.subplots(nrows=len(high_correlation), ncols=1, figsize=(6, 4*len(high_correlation)), constrained_layout=True)
                for i, corr_pair in enumerate(high_correlation):
                    ax2[i].scatter(numericals[corr_pair[0]], numericals[corr_pair[1]], c='blue')
                    ax2[i].set_xlabel(corr_pair[0])
                    ax2[i].set_ylabel(corr_pair[1]) 

                fig2.suptitle('Highly Correlated Variables')

            # Plot distribution histograms
            fig3, ax3 = plt.subplots(nrows=n_graphs, ncols=1, figsize=(8, 3*n_graphs), constrained_layout=True)
            colors = plt.rcParams["axes.prop_cycle"]()
            for i, column_name in enumerate(numericals):
                c = next(colors)["color"] # Iterate colors
                ax3[i].hist(numericals[column_name], bins=30, color=c) # Draw histogram
                m = numericals[column_name].mean() # Calculate mean
                ax3[i].axvline(x=m, color='red', linestyle='--', linewidth=2, label='Avg') # Draw mean line
                ax3[i].text(x=m, y=ax3[i].get_ylim()[1]*0.95, s='Mean: '+str(round(m, 2)), color='red', ha='left') # Place text with mean value close to mean line
                ax3[i].set_xlabel(column_name)
                ax3[i].set_ylabel('Frequency') 

            fig3.suptitle('Numeric Variable Distributions')

        if not categoricals.empty:
            fig4, ax4 = plt.subplots(nrows=len(categoricals.columns), ncols=1, figsize=(6, 4*len(categoricals.columns)), constrained_layout=True)
            for i, ax in enumerate(ax4):
                categoricals[categoricals.columns[i]].value_counts().iloc[:10].plot.barh(ax=ax, x=categoricals.columns[i], y='Count', rot=0, title='Categorical Variable Distributions')

        plt.show()
        
    def get_X_and_y(self):
        """
        Return dependent (X) and target (y) variables.
        """
        self.dataframe.drop_duplicates(inplace=True) # Remove duplicates
        X = self.dataframe.drop(columns=[self.target_name])
        y = self.dataframe[self.target_name]
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
    

    def evaluate(self, pipelines: list, X, y, test_type: str):
        """
        This function receives the pipelines to be crossvalidated, takes the one with the highest score and applies the appropiate evaluation method
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

        for i, pipeline in enumerate(pipelines):
            scores = cross_val_score(pipeline, X, y, scoring=test_type, cv=5)
            mean = np.mean(scores)
            print(f'Pipeline {i}:', scores, f'Mean score: {mean}')

            if mean > highscore[1]:
                highscore[0] = pipeline
                highscore[1] = mean
                highscore[2] = i
        
        highscore[0].fit(X_train, y_train)
        y_pred = highscore[0].predict(X_test)

        if test_type in regression:
            print(f'\nPrinting results for pipeline {highscore[2]}:')
            print('MSE:', mean_squared_error(y_test, y_pred))
            print('R-Squared:', r2_score(y_test, y_pred))
            print('MAPE:', mean_absolute_percentage_error(y_test, y_pred))
            print('MAE:', mean_absolute_error(y_test, y_pred))
        elif test_type in classification:
            print(f'\nPrinting results for pipeline {highscore[2]}:')
            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0].classes_)
            disp.plot()

            print('Accuracy:', accuracy_score(y_test, y_pred))
            print('Precission:', precision_score(y_test, y_pred)) 
            print('Recall:', recall_score(y_test, y_pred))
            print('F1 score:', f1_score(y_test, y_pred))
        else:
            print('Couldn\'t identify a proper evaluation method')