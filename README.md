# Machine-Learning-Final-Project----Vitellaro-Salvatore
CS 4200

This dataset is information on alternative-fuel stations. I am interested in predicting the location and quantity of new stations in the future.
I think that the most relevant column types are dates, state, ZIP code, and latitude and longitude.

!["Count by Year"](./Figure_1.png "Count by Year")


!["Count by Year"](./Figure_2.png "Count by Year")


Most of the columns dropped were either mostly empty or had no bearing on the metrics being considered.
State was hot-encoded because there are many states.

!["Residual Plot"](./residuals.png "Residual Plot")

Here it can be seen that the residuals are not evenly distributed. A linear regression model is probably not the best model for this data.
This is not surprising considering that population centers, where one might resaonably expect more alternative fuel stations, are not linearly distributed across the US.
