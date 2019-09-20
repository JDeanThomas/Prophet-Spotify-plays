import fbprophet
import pandas as pd
from scipy.stats import boxcox
from scipy.special import inv_boxcox

# Create DF from API query
df = datasets['plays_daily']

df.shape
df.dtypes

df['ds'] = df['date']
df['y'] = df['value']

df.set_index('date')

# Apply Box-Cox Transform to value column and assign to new column y
df['y'], lam = boxcox(df['value'])

model = fbprophet.Prophet()
model.fit(df)

future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

model.plot(forecast);
model.plot_components(forecast);

# Apply inverse Box-Cox transform to specific forecast columns
forecast[['yhat', 'yhat_upper', 'yhat_lower']] = forecast[['yhat', 
          'yhat_upper', 'yhat_lower']].apply(lambda x: inv_boxcox(x, lam))