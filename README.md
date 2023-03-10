# Analysis_And_Generation_Model_ML

__BEFORE READING THIS REPOSITORY IT IS RECOMMENDED TO START FROM:__

https://github.com/insidbyte/Analysis_and_processing

***I have in fact decided to generate a custom vocabulary to train the model and it would be appropriate to look at the repository code.***

___OPTIONS:___
 
    1)-GENERATE MODEL

    2)-TEST WITH HYPERPARAMETER TUNING

    3)-PLOT WITH TFIDF VECTORIZER AND SVD TRUNCATED REDUCTION

### Menù

__Starting the ModelsGenerator.py file from the terminal it will appear:__

![Screenshot](myScripts/OUTPUTS/menu.png)

#  ___OPTION 1:___
### Model generation:
***I decided to use tfidf and support vector machine because they are highly suitable for text processing and***
***support vector machine with the linear kernel is highly suitable for classifications based on two classes as***
***in our case: positive and negative***

![Screenshot](myScripts/OUTPUTS/generator.png)

# Kaggle IMDb dataset example:

![Screenshot](myScripts/OUTPUTS/prova.png)

### ***I created a Client in Angular to send requests to a Python Server*** 

# CLIENT: 

![Screenshot](myScripts/OUTPUTS/client.png)

# SERVER:

![Screenshot](myScripts/OUTPUTS/server.png)

# RESPONSE FROM THE SERVER:

![Screenshot](myScripts/OUTPUTS/response.png)

# ANOTHER EXAMPLE:

![Screenshot](myScripts/OUTPUTS/prova2.png)

![Screenshot](myScripts/OUTPUTS/response2.png)

#  ___OPTION 2:___
### Test hyperparameters with gridsearchCV and tfidf vectorizer:

***A good way to automate the test phase and save time searching for the best parameters to***
***generate the most accurate model possible is to use GrisearchCV made available by scikit-learn***
***The code in ModelsGenerator.py must be customized based on the dataset to be analyzed***

### ***WARNING !!***
### ***If we don't study the scikit-learn documentation we could start infinite analyzes***
### ***so it is always advisable to know what we are doing***

### Link scikit-learn: https://scikit-learn.org/

***Input:***

![Screenshot](myScripts/OUTPUTS/grid.png)


***Output:***

![Screenshot](myScripts/OUTPUTS/grid2.png)



#  ___OPTION 3:___

***Input:***

![Screenshot](myScripts/OUTPUTS/plot.png)

***This option is experimental, the reduction is not applied to model training because it***
***generates too few components and RAM memory (8GB) of my PC is not enough to generate***
***more components even if the results are interesting!***

***Output:***

![Screenshot](myScripts/OUTPUTS/TSVD_result.png)

# CONCLUSION:

***We got satisfactory results and generated a fairly accurate***
***model this repository will be updated over time***

***For info or collaborations contact me at: u.calice@studenti.poliba.it***
