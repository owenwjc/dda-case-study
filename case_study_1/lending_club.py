#!/usr/bin/env python
# coding: utf-8

# # Setup and Description of Data

# In[1]:


import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# In[2]:


df = pd.read_csv("../data/loans_full_schema.csv") # load in data csv from data folder


# The dataset represents a snapshot of peer to peer loan data from 3 months in 2018. It includes detailed borrower and loan information for 10000 unique loans in various stages of repayment.<p>
# The dataset does have some issues mainly due to its relativly small size. A lot of the data is not particularly useable because it doesn't contain more information.<p>
# - If it covered a larger length of time there are more trends you could analyze. For example, you could analyze how seasons influence home improvement borrowing. After a heatwave during summer or a snowstorm in the winter you would see more borrowing over the next year.<p>
# - The location information isn't fine enough. You can't control for cost of living with just the state.<p>
# - The loan data has no ID so you can't track an individual loan over time.<p>
# - With finer date information you could see if certain loans are taken out on certain days. Like if credit card loans are taken out around when bills are due.<p>
# 
# Basically the dataset includes a lot of information that can't be used because of its size.

# # Prep Tables

# In[3]:


# use joint income if it exists
df['annual_income_joint'].fillna(df['annual_income'], inplace=True)
df['debt_to_income_joint'].fillna(df['debt_to_income'], inplace=True)

df.drop(columns=['annual_income', 'debt_to_income'], inplace=True)
df.rename(columns={'annual_income_joint':'annual_income', 'debt_to_income_joint': 'debt_to_income'}, inplace=True)


# In[4]:


credit_share = df['total_credit_utilized'] / (df['total_credit_limit'])             # share of credit limit that is used
credit_ratio = df['total_credit_limit'] / (df['annual_income'])                     # ratio between annual income and credit limit
interest_share = df['paid_interest'] / (df['paid_interest'] + df['paid_principal']) # share of current loan payments that was interest
paid_percentage = df['paid_principal'] / df['loan_amount']                          # how much of the loan is paid off
loan_ratio = (df['loan_amount'] / df['annual_income'])                              # ratio of loan amount to annual income

# insert ratios back into table
df.insert(0, 'interest_share', interest_share.fillna(0))
df.insert(0, 'credit_share', credit_share.fillna(0))
df.insert(0, 'credit_ratio', credit_ratio)
df.insert(0, 'paid_percentage', paid_percentage)
df.insert(0, 'loan_ratio', loan_ratio.fillna(0))


# In[5]:


bins = [0, 30000, 60000, 120000, 240000, 480000, 3840000] # income bins for histogram
binlabels = ['0-30000', '30000-60000', '60000-120000', '120000-240000', '240000-480000', '480000-3840000']

# sort incomes into bins and insert the bin labels into the table
df.insert(0,'binned', pd.cut(df['annual_income'], bins, labels=binlabels))

# create tables for bar graphs
loan_bar_source = df.groupby(['binned', 'grade'], as_index=False).count()
home_bar_source = df.groupby(['binned', 'homeownership'], as_index=False).count()
# normalize and insert homeownership data in bar graph table
home_bar_source.insert(2, 'normalized', home_bar_source.groupby('binned').transform(lambda x: (x) / (x.sum()))['debt_to_income'])

# calculate average debt to income
mean_dti = df['debt_to_income'].mean()

# general borrower and loan information tables
borrower_info = df[['annual_income', 'debt_to_income', 'credit_share']]
loan_info = df[['interest_rate', 'interest_share', 'paid_percentage']]


# # Visualizations

# In[6]:


fig = px.scatter_matrix(borrower_info, width=1000, height=1000)
fig.show()


# This is a scatterplot matrix looking at various information related to the borrowers. It shows a slight positive correlation between debt to income and how much credit a borrower uses. It shows negative correlations between debt to income/credit use and annual income.<p>
# This seems intuitive. Generally, you might assume that someone who carries more debt uses more of their available credit. You can also assume that as a persons income increases their debts decrease and their available credit increases faster than their utilization.

# In[7]:


fig = px.scatter_matrix(loan_info, width=1000, height=1000)
fig.show()


# Here is another scatterplot matrix showing various information about the loans. It shows a very strong negative exponential correlation between how much of a loan is paid off and how much of those payments goes to interest.<p>
# The other graphs don't have as strong of correlations but they show that the interest rate definitely affects how much payment goes towards interest, and that it is relatively harder to pay off a loan with a higher interest rate.<p>
# 
# There is a more detailed look at the highly correlated share and perecentage graph further down.

# In[8]:


purposes = ['credit_card', 'debt_consolidation', 'home_improvement', 'major_purchase', 'medical', 'small_business']

fig = px.scatter(df[df['loan_purpose'].isin(purposes)], x='paid_percentage', y='loan_amount', color='loan_purpose', marginal_y='box', marginal_x='box', size='annual_income', size_max=25, width=1000, height=1000)

fig.show()


# Here's a look at the general status of loans for the 6 most popular purposes. The size of the dots represents the borrowers annual income and the color shows the loans purpose.<p>
# Unsurprisingly loans are fairly evenly distributed across all amounts and there is no obvious signs that larger loans are harder to pay off.<p>
# For me what was unexpected from this visualization was that medical loans were the smallest on average. I though that they would be the highest but they are even smaller than credit card loans. This may just be due to the small sample size but it was surprising to see.

# In[9]:


fig = px.scatter(df, x='debt_to_income', y='credit_share', marginal_x='box', marginal_y='box', color='grade', width=1000, height=1000)
fig.show()


# This is a closeup of the scatterplot from above showing the positively correlated borrower dti to credit usage. It is color coded by the loan grade and shows that as debt share and credit share go up the loan grade increases.

# In[10]:


fig = go.Figure()

grades = df['grade'].unique()
grades.sort()

for grade in grades:
    fig.add_trace(go.Violin(x=df['grade'][df['grade'] == grade],
                            y=df['interest_share'][df['grade'] == grade],
                            name=grade,
                            box_visible=True,
                            meanline_visible=True))

fig.update_layout(width=1000, height=1000)
fig.update_layout(xaxis_title='Loan Grade', yaxis_title='Interest Payment Percentage', legend_title='Loan Grade')
fig.show()


# Since I already calculated the percentage of loan payment that went to interest I wanted to see how that was affected by loan grade.<p>
# This graph shows how much interest each loan grade actually makes. So while the iterest rate of D graded loans is around 20 percent the borrowers are paying between 40 and 60 percent towards interest.

# In[11]:


bins = loan_bar_source['binned'].unique().astype(str)
fig = go.Figure(data = [
    go.Bar(name='A', x=bins, y=loan_bar_source['interest_share'][loan_bar_source['grade'] == 'A']),
    go.Bar(name='B', x=bins, y=loan_bar_source['interest_share'][loan_bar_source['grade'] == 'B']),
    go.Bar(name='C', x=bins, y=loan_bar_source['interest_share'][loan_bar_source['grade'] == 'C']),
    go.Bar(name='D', x=bins, y=loan_bar_source['interest_share'][loan_bar_source['grade'] == 'D']),
    go.Bar(name='E', x=bins, y=loan_bar_source['interest_share'][loan_bar_source['grade'] == 'E']),
    go.Bar(name='F', x=bins, y=loan_bar_source['interest_share'][loan_bar_source['grade'] == 'F']),
    go.Bar(name='G', x=bins, y=loan_bar_source['interest_share'][loan_bar_source['grade'] == 'G']),
])

fig.update_layout(barmode='group', width=1000, height=1000)
fig.update_layout(xaxis_title='Income Band', yaxis_title='Number of Loans', legend_title='Loan Grade')
fig.show()


# This is a bar chart showing loan grade makeup for differnt income bands. It is a fairly normal distribution with an unsurprising shift in loan grade from low to high income.

# In[12]:


fig = go.Figure()

for band in binlabels:
    fig.add_trace(go.Violin(x=df['binned'][(df['binned'] == band) & (df['debt_to_income'] > mean_dti)],
                            y=df['loan_amount'][(df['binned'] == band) & (df['debt_to_income'] > mean_dti)],
                            legendgroup='high debt_to_income',
                            name=band,
                            side='negative',
                            line_color='blue'))

    fig.add_trace(go.Violin(x=df['binned'][(df['binned'] == band) & (df['debt_to_income'] <= mean_dti)],
                            y=df['loan_amount'][(df['binned'] == band) & (df['debt_to_income'] <= mean_dti)],
                            legendgroup='low debt_to_income',
                            name=band,
                            side='positive',
                            line_color='orange'))
                            
fig.update_traces(meanline_visible=True)                            
fig.update_layout(violingap=0, violinmode='overlay', width=1000, height=1000)
fig.update_layout(xaxis_title='Income Band', yaxis_title='Loan Amount', legend_title='Debt to Income')
fig.show()


# And this is the makeup of the loans for each income band split between high debt to income and low debt to income. The blue represents high dti borrowers and shows a higher loan amount on average compared to low dti borrowers in the same income band.<p>
# As income increases the loans shift to higher values. But surprisingly the relative distribution of loan amounts reverses around an annual income of 120000 with the peak of the distribution moving to a much higher loan amount.

# In[13]:


bins = home_bar_source['binned'].unique().astype(str)
fig = go.Figure(data = [
    go.Bar(name='MORTGAGE', x=bins, y=home_bar_source['normalized'][home_bar_source['homeownership'] == 'MORTGAGE']),
    go.Bar(name='RENT', x=bins, y=home_bar_source['normalized'][home_bar_source['homeownership'] == 'RENT']),
    go.Bar(name='OWN', x=bins, y=home_bar_source['normalized'][home_bar_source['homeownership'] == 'OWN']),
])

fig.update_layout(barmode='relative', width=1000, height=1000)
fig.update_layout(xaxis_title='Income Band', yaxis_title='Percent', legend_title='Home Ownership Status')
fig.show()


# I thought the shift around 120000 might be due to homeownership as that represents a source of large costs that might not be present for borrowers in lower income brackets. While the graph does show a decrease in renters and proportionate rise in mortgages I doubt it is the only source for the flipped distribution.

# In[14]:


fig = px.scatter(df, x='interest_share', y='paid_percentage', marginal_x='box', marginal_y='box', color='grade', width=1000, height=1000)
fig.show()


# This graph shows the strongest correlation that I found in the data. Showing how the more a borrower pays towards interest the farther they are from fully paying off their loan. It also shows the bands that each loan grade occupies.

# # Model

# In[15]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import math


# In[16]:


def custom_round(x, base=5):
    return base * round(float(x)/base)

def runModel(forestdf, leaf_samples):
    y = pd.DataFrame(forestdf['interest_rate'])
    X = forestdf.drop(columns=['interest_rate'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)

    model = RandomForestRegressor(n_estimators=100, min_samples_leaf=leaf_samples)
    model.fit(X_train, y_train.squeeze())

    prediction = model.predict(X_test)
    print("Mean Absolute Error = ", abs((y_test.squeeze() - prediction)).mean())
    print("Root Mean Squared Error = ", math.sqrt(mean_squared_error(y_test, prediction)))
    print("Train Accuracy = ", model.score(X_train, y_train))
    print("Test Accuracy = ", model.score(X_test, y_test))
    return model, X

def linearModel(forestdf):
    y = pd.DataFrame(forestdf['interest_rate'])
    X = forestdf.drop(columns=['interest_rate'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)

    model = LinearRegression()
    model.fit(X_train, y_train.squeeze())

    prediction = model.predict(X_test)
    print("Mean Absolute Error = ", abs((y_test.squeeze() - prediction)).mean())
    print("Root Mean Squared Error = ", math.sqrt(mean_squared_error(y_test, prediction)))
    print("Train Accuracy = ", model.score(X_train, y_train))
    print("Test Accuracy = ", model.score(X_test, y_test))
    return model, X


# Since we want to make a perfect model we can choose the feature set that is best described by interest rate.<p>
# To start I built a Random Forest Regressor using the features loan amount, term, and installment. That way the model can figure out how to calculate interest rate directly.

# In[17]:


keepColumns = ['loan_amount', 'term', 'interest_rate', 'installment']
forestdf = df[keepColumns]


# In[18]:


model, X = runModel(forestdf, 1)
pd.Series(model.feature_importances_, index=X.columns).plot(kind='bar')


# Obviously these results are too good. The average error is .18 percent. You can't expect to be given the information directly derived from the interest rate and told to solve for the interest rate.<p>
# So lets take a more naive approach.<p>
# I chose the following features and performed the same random forest regression:<p>
# features = ['credit_share', 'debt_to_income', 'loan_ratio', 'interest_share', 'earliest_credit_line', 'inquiries_last_12m', 'term', 'interest_rate', 'issue_month', 'account_never_delinq_percent', 'homeownership']<p>
# I wanted features that would reflect the borrower and I included payment information in the form of interest_share since I assumed the data might be provided as a snapshot of the current loan payment progress.<p>
# I chose credit_share, earliest_credit_line, inquiries_last_12m, account_never_delinq_percent as a credit score stand in. I thought home ownership status might be important so I included it and one hot encoded it. I also thought that the month might have had an influence on the interest rate so I also one hot encoded that. 
# The last bits of information I included was the term chosen for the loan, the ratio between the loan and the borrowers annual income, and the borrowers debt to income.<p>
# 
# To avoid overfitting I rounded some of the finer numerical data to make it more categorical.

# In[19]:


keepColumns = ['credit_share', 'debt_to_income', 'loan_ratio', 'interest_share', 'earliest_credit_line', 'inquiries_last_12m', 'term', 'interest_rate', 'issue_month', 'account_never_delinq_percent', 'homeownership']
forestdf = df[keepColumns]

# one hot encoding
forestdf = forestdf.join(pd.get_dummies(forestdf.homeownership, prefix='homeownership'))
forestdf = forestdf.join(pd.get_dummies(forestdf.issue_month, prefix='issue_month'))
forestdf.drop(columns=['homeownership', 'issue_month'], inplace=True)

forestdf['credit_share'] = forestdf['credit_share'].apply(lambda x: custom_round(x, base=0.05))
forestdf['debt_to_income'] = forestdf['debt_to_income'].apply(lambda x: custom_round(x, base=1))
forestdf['loan_ratio'] = forestdf['loan_ratio'].apply(lambda x: custom_round(x, base=0.01))


# In[20]:


model, X = runModel(forestdf, 10)
pd.Series(model.feature_importances_, index=X.columns).plot(kind='bar')


# The results are still a bit suspicious and looking at the feature importance we can see why. The interest share is too descriptive of the interest rate so if we want a truly naive algorithm to find interest rate it will have to be left out.

# In[21]:


keepColumns = ['credit_share', 'debt_to_income', 'loan_ratio', 'earliest_credit_line', 'inquiries_last_12m', 'term', 'interest_rate', 'issue_month', 'account_never_delinq_percent', 'homeownership']
forestdf = df[keepColumns]

# one hot encoding
forestdf = forestdf.join(pd.get_dummies(forestdf.homeownership, prefix='homeownership'))
forestdf = forestdf.join(pd.get_dummies(forestdf.issue_month, prefix='issue_month'))
forestdf.drop(columns=['homeownership', 'issue_month'], inplace=True)

forestdf['credit_share'] = forestdf['credit_share'].apply(lambda x: custom_round(x, base=0.05))
forestdf['debt_to_income'] = forestdf['debt_to_income'].apply(lambda x: custom_round(x, base=1))
forestdf['loan_ratio'] = forestdf['loan_ratio'].apply(lambda x: custom_round(x, base=0.01))


# In[22]:


model, X = runModel(forestdf, 25)
pd.Series(model.feature_importances_, index=X.columns).plot(kind='bar')


# With a bit of hyperparameter tuning I got 31 percent accuracy and a mean absolute error of  just over 3 percent. The overal accuracy was scored to be 31 percent. So its a much more realistic result given the nature of the problem. Since there are 7 grades each spanning about 3 percent the model is fairly adept at correctly identifying a loans grade. That being said the random forest still slightly overfits to the training data.

# In[80]:


keepColumns = ['credit_share', 'debt_to_income', 'loan_ratio', 'earliest_credit_line', 'inquiries_last_12m', 'term', 'interest_rate', 'issue_month', 'account_never_delinq_percent', 'homeownership']
forestdf = df[keepColumns]

# one hot encoding
forestdf = forestdf.join(pd.get_dummies(forestdf.homeownership, prefix='homeownership'))
forestdf = forestdf.join(pd.get_dummies(forestdf.issue_month, prefix='issue_month'))
forestdf.drop(columns=['homeownership', 'issue_month'], inplace=True)

forestdf['credit_share'] = forestdf['credit_share'].apply(lambda x: custom_round(x, base=0.05))
forestdf['debt_to_income'] = forestdf['debt_to_income'].apply(lambda x: custom_round(x, base=1))
forestdf['loan_ratio'] = forestdf['loan_ratio'].apply(lambda x: custom_round(x, base=0.01))


# In[81]:


model, X = linearModel(forestdf)


# Putting the same data through a linear regression model gives a lower accuracy but doesn't have the same overfitting issues as the random forest.

# With more time I would do a proper analysis of all the features then I would do a proper run through tuning the hyperparameters of the random forest regressor. If further accuracy were needed I would build an neural network model.
