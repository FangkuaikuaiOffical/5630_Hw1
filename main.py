import pandas
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# The data was preprocessed using excel and headers
# were recorded for all discrete data columns.
nominal_columns = ['Column2', 'Column3', 'Column4', 'Column5', 'Column7',
                   'Column8', 'Column9', 'Column10', 'Column11', 'Column12',
                   'Column13', 'Column14', 'Column15', 'Column16', 'Column20',
                   'Column21', 'Column22', 'Column23', 'Column24', 'Column26',
                   'Column27', 'Column28', 'Column29', 'Column30', 'Column32',
                   'Column33', 'Column34', 'Column35', 'Column36', 'Column37',
                   'Column38', 'Column39', 'Column41', 'Column42']

# Import data and do OHE.
data = pandas.read_csv("data/income_data.csv")
test = pandas.read_csv("data/income_test.csv")
data_onehot = pandas.get_dummies(data, columns=nominal_columns, drop_first=True)
test_onehot = pandas.get_dummies(test, columns=nominal_columns, drop_first=True)

# Since the "Grandchild <18 ever marr not in subfamily" value was not included in the test data set,
# it was removed from the original data. Otherwise, an error will be reported during verification.
data_X = data_onehot.drop(columns=['Column42_ 50000+.', 'Column23_ Grandchild <18 ever marr not in subfamily'])
data_Y = data_onehot['Column42_ 50000+.']
test_X = test_onehot.drop(columns='Column42_ 50000+.')
test_Y = test_onehot['Column42_ 50000+.']

# Generate decision trees of different depths and compare them with tests.
for depth in range(2, 21):
    dtree = DecisionTreeClassifier(max_depth=depth)
    dtree.fit(data_X, data_Y)
    predictions = dtree.predict(data_X)
    acc = accuracy_score(data_Y, predictions)
    predictions2 = dtree.predict(test_X)
    acc2 = accuracy_score(test_Y, predictions2)
    print('The accuracy when depth set ' + str(depth) + ' is ' + str(acc)
          + ' for it self and ' + str(acc2) + ' for test ')
