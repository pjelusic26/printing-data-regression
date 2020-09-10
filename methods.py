import pandas as pd
import numpy as np

# For splitting data, feature importance and metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import preprocessing

# For neural network
import keras.activations
import keras_metrics as km
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

# TensorFlow docs
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

class RegressionNeograf:
    
    @staticmethod
    def DataFrameImport(filepath):
        """Reads Excel file as Pandas DataFrame

        Arguments:
            filepath {string} -- path and name of file

        Returns:
            Pandas DataFrame
        """
        return pd.read_excel(filepath, decimal = '.')
    
    @staticmethod
    def DataFrameMerge(df1, df2):
        """Merges two Pandas DataFrames

        Arguments:
            df1, df2 {pandas.df} -- DataFrames to merge

        Returns:
            Pandas DataFrame
        """
        return pd.merge(df1, df2, how = 'outer')
    
    @staticmethod
    def DataFrameDrop(df):
        """Makes various Pandas DataFrame drops. These drops are unique for this project.

        Arguments:
            df {pandas.df} -- DataFrame to process

        Returns:
            Pandas DataFrame
        """
        df = df.drop(['Unnamed: 0'], axis = 1)
        df = df.dropna(axis = 'rows')

        # Remove all data points larger than an 4-hour shift
        df = df[df['Sekunde'] < 14400]

        df = df[df.Sekunde != 0]
        df = df[df.proizvodi != 0]
        df = df[df.skart != 0]

        df = df.drop(['karton_gt', 'boje_b', 'lak_uljni_sjajni', 'naziv',
                 'lak_vododisperzivni', 'lak_vododisperzivni_uljni_mat',
                 'lak_vododisperzivni_mat', 'lak_vododisperzivni_sjajni',
                 'karton_gc1', 'karton_gc2', 'karton_gd', 'lak_nacin', 'broj_prolaza', 'naklada'], 
                axis = 1)

        return df
    
    @staticmethod
    def DataFramePreProcess(df):
        """Makes various Pandas DataFrame changes. These changes are unique for this project.

        Arguments:
            df {pandas.df} -- DataFrame to process

        Returns:
            Pandas DataFrame
        """
        df['naklada_final'] = df['naklada'] / df['kutija_na_tiskarskom_arku']
        df['production_ratio'] = df['naklada_final'] / df['proizvodi']
        df['production_speed'] = df['Sekunde'] / df['proizvodi']
        
        df = df[df['production_ratio'] < 1.25]
        df = df[df['production_ratio'] > 0.80]

        dfResult = df.drop(['naklada_final', 'production_ratio', 'production_speed'], 
                       axis = 1)

        return dfResult
    
    @staticmethod
    def DataFrameNormalize(df):
        """Normalizes input parameters of Pandas DataFrame

        Arguments:
            df {pandas.df} -- DataFrame to process

        Returns:
            Normalized Pandas DataFrame
        """
        result = df.copy()
        for feature_name in df.columns:
            max_value = df[feature_name].max()
            min_value = df[feature_name].min()
            result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    
        # Pop column from DF
        popCol = df.pop('Sekunde')

        # Return popped column to DF
        df['Sekunde'] = popCol
        result['Trajanje'] = popCol

        result.drop(['Sekunde'], axis = 1, inplace = True)
        
        return df, result

    @staticmethod
    def DataSplitting(dfNorm, testSize, randState):
        """Splits DataFrame to train and test / inputs and output

        Arguments:
            dfNorm {pandas.df} -- DataFrame to process
            testSize {float} -- percentage of DF for testing
            randState {int} -- controls the key of randomization

        Returns:
            train_X -- training input parameters
            test_X -- test input parameters
            train_y -- training target parameter
            test_y -- test target parameter
        """
        # Split data into train and test
        train_X, test_X = train_test_split(dfNorm, test_size = testSize, random_state = randState)

        # Create target (output)
        train_y = train_X[['Trajanje']]
        test_y = test_X[['Trajanje']]

        # Drop target (output) column from input datasets
        train_X.drop(columns = ['Trajanje'], inplace = True)
        test_X.drop(columns = ['Trajanje'], inplace = True)

        return train_X, test_X, train_y, test_y
    
    @staticmethod
    def InputOutputNormalize(dfTrain, dfTest):
        """Splits DataFrame to train and test / inputs and output

        Arguments:
            dfNorm {pandas.df} -- DataFrame to process
            testSize {float} -- percentage of DF for testing
            randState {int} -- controls the key of randomization

        Returns:
            train_X -- training input parameters
            test_X -- test input parameters
            train_y -- training target parameter
            test_y -- test target parameter
        """
        resultTrain = dfTrain.copy()
        for feature_name in dfTrain.columns:
            max_value = dfTrain[feature_name].max()
            min_value = dfTrain[feature_name].min()
            resultTrain[feature_name] = (dfTrain[feature_name] - min_value) / (max_value - min_value)
            
        resultTest = dfTest.copy()
        for feature_name in dfTest.columns:
            max_value = dfTest[feature_name].max()
            min_value = dfTest[feature_name].min()
            resultTest[feature_name] = (dfTest[feature_name] - min_value) / (max_value - min_value)
            
        # Pop column from DF
        popColTrain = dfTrain.pop('Sekunde')
        
        # Return popped column to DF
        dfTrain['Sekunde'] = popColTrain
        resultTrain['Trajanje'] = popColTrain   
        
        # Pop column from DF
        popColTest = dfTest.pop('Sekunde')
        
        # Return popped column to DF
        dfTest['Sekunde'] = popColTest
        resultTest['Trajanje'] = popColTest  
        
        # Assign values to variables
        train_X = resultTrain
        test_X = resultTest

        # Create target (output)
        train_y = train_X[['Trajanje']]
        test_y = test_X[['Trajanje']]

        # Drop target (output) column from input datasets
        train_X.drop(columns = ['Trajanje'], inplace = True)
        test_X.drop(columns = ['Trajanje'], inplace = True)        
      
        return train_X, test_X, train_y, test_y
    
    @staticmethod
    def NeuralNet(dataSplits, layerNeurons, opt, lossFunc, epochNum, batchSize, valSplit):
        """Runs neural network

        Arguments:
            dataSplits {ndarray} -- train and test samples
            layerNeurons {int} -- number of neurons in neural network layers
            opt {string} -- Keras optimizer of choice
            lossFunc {string} -- Keras loss function of choice
            epochNum {int} -- number of epochs
            batchSize {int} -- size of batch
            valSplit {int} -- validation split

        Returns:
            test_y {ndarray} -- array with real and predicted values
        """
        train_X = dataSplits[0]
        test_X = dataSplits[1]
        train_y = dataSplits[2]
        test_y = dataSplits[3]

        # Create model
        model = Sequential()

        # Get number of columns in training data
        n_cols = train_X.shape[1]

        # Add model layers
        model.add(Dense(layerNeurons, activation = 'relu', input_shape = (n_cols,)))
        model.add(Dense(layerNeurons, activation = 'relu', input_shape = (n_cols,)))
        model.add(Dense(1))
    
        # Compile model using mse as a measure of model performance
        model.compile(optimizer = opt, loss = lossFunc)
    
        # Train model
        model.fit(train_X, train_y, epochs = epochNum, batch_size = batchSize, validation_split = valSplit, verbose = 0)

        # Save predictions
        y_pred = model.predict(test_X)
        # r2Score = round(r2_score(test_y, y_pred), 3)
    
        # Add predictions to test_y as a new column (for the graph below)
        test_y['Predictions'] = y_pred
        
        return test_y
    
    @staticmethod
    def NeuralNetErrorMetrics(resultsDF):
        """Calculates prediction percentage errors

        Arguments:
            resultsDF {pandas.df} -- DF to analyse

        Returns:
            diff {pandas.df} -- real percentage error
            diffProduction {pandas.df} -- production percentage error
        """
        # Turning columns to values
        testY = resultsDF['Trajanje'].values
        preds = resultsDF['Predictions'].values

        resultsDF['minValue'] = resultsDF.min(axis = 1)
        minValue = resultsDF['minValue'].values

        # Calculate difference between real and predicted values
        diff = np.abs((testY - preds) / testY) * 100
        diffProduction = np.abs((testY - preds) / minValue) * 100
        # test_y['diff'] = diff
        # test_y['diffProduction'] = diffProduction

        # Put values in DF
        diff = pd.DataFrame(data = diff, columns = ['Error %'])
        diffProduction = pd.DataFrame(data = diffProduction, columns = ['Error %'])
        
        return diff, diffProduction

    @staticmethod
    def SaveToExcel(df, filepath):
        """Saves Pandas DataFrame as Excel document

        Arguments:
            df {pandas.df} -- DF to save
            filepath {string} -- path and name

        Returns:
            Excel document
        """
        return df.to_excel(filepath)