import BlackBoxAuditing as BBA
# import machine learning technique
from BlackBoxAuditing.model_factories import SVM, DecisionTree, NeuralNetwork


"""
Using your own dataset
"""
# load your own data
datafile = '../../data/adult.csv'
data = BBA.load_from_file(datafile)

# initialize the auditor and set parameters
auditor = BBA.Auditor()
auditor.ModelFactory= DecisionTree

# call the auditor with the data
auditor(data, output_dir="adult_output")

# find contexts of discrimination in dataset
auditor.find_contexts("race", output_dir="adult_output")
