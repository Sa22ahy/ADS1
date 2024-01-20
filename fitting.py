import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import scipy.optimize as opt
import numpy as np

import errors as err


def read_data(file_paths, selected_country, start_year, end_year):
    dataframes_list = []

    for path in file_paths:
        file_name = path.split('/')[-1].split('.')[0]
        df = pd.read_csv(path, skiprows=4)
        df = df.rename(columns={'Country Name': 'Country'})
        df = df.set_index('Country')
        df_selected_country = df.loc[selected_country, str(start_year):str(end_year)].transpose().reset_index()
        df_selected_country = df_selected_country.rename(columns={'index': 'Year', selected_country: file_name})
        dataframes_list.append(df_selected_country)

    # Concatenate all DataFrames based on the 'Year' column
    result_df = pd.concat(dataframes_list, axis=1)

    # Replace null values with the mean of each column
    result_df = result_df.apply(lambda col: col.fillna(col.mean()))

    return result_df


def exp_growth(t, scale, growth):
        """ Computes exponential function with scale and growth as free parameters """
   
        f = scale * np.exp(growth * t)
        return f
    

def fit_and_plot_growth_model(technology_df, selected_country, model_function, output_filename):
    initial_guess = [1.0, 0.02]
    popt, pcovar = opt.curve_fit(model_function, technology_df["Year"], technology_df["technology exports"], p0=initial_guess, maxfev=10000)
    print("Fit parameters:", popt)

    # Create a new column with the fitted values
    technology_df["pop_exp"] = model_function(technology_df["Year"], *popt)

    # Plot
    plt.figure()
    plt.plot(technology_df["Year"], technology_df["technology exports"], label="data")
    plt.plot(technology_df["Year"], technology_df["pop_exp"], label="fit")
    plt.legend()
    plt.title(f"Data Fit attempt for {selected_country}")
    plt.show()

    # Call function to calculate upper and lower limits with extrapolation
    # Create an extended year range
    years = np.linspace(2007, 2030)
    pop_exp_growth = model_function(years, *popt)
    sigma = err.error_prop(years, model_function, popt, pcovar)
    low = pop_exp_growth - sigma
    up = pop_exp_growth + sigma

    plt.figure()
    plt.title(f"Technology exports of {selected_country} in 2030")
    plt.plot(technology_df["Year"], technology_df["technology exports"], label="data")
    plt.plot(years, pop_exp_growth, label="fit")
    # Plot error ranges with transparency
    plt.fill_between(years, low, up, alpha=0.3, color="y", label="95% Confidence Interval")
    plt.legend(loc="upper left")
    plt.xlabel("Year")
    plt.ylabel("Technology exports")
    # Set the dpi parameter to 300 when saving the plot
    plt.savefig(f'{output_filename}.png', dpi=300)
    plt.show()

    # Predict future values
    pop_2030 = model_function(np.array([2030]), *popt)
    # Assuming you want predictions for the next 10 years
    sigma_2030 = err.error_prop(np.array([2030]), model_function, popt, pcovar)
    print(f"Technology exports in 2030 for {selected_country}: {pop_2030 / 1.0e6} Mill.")

    # For the next 10 years
    print(f"Technology exports for {selected_country} in the next 10 years:")
    for year in range(2024, 2034):
        print(f"{year}: {model_function(year, *popt) / 1.0e6} Mill.")


# Example usage
file_paths = ['technology exports.csv']
selected_countries = ["Australia", "China", "Pakistan", "India"]
start_year = 2007
end_year = 2022

for country in selected_countries:
    technology_df = read_data(file_paths, country, start_year, end_year)
    technology_df["Year"] = pd.to_numeric(technology_df["Year"], errors='coerce')
    technology_df["technology exports"] = pd.to_numeric(technology_df["technology exports"], errors='coerce')

    fit_and_plot_growth_model(technology_df, country, exp_growth, f'technology_exports_{country.lower()}')
