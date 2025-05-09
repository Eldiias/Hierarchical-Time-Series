"""
"""

import pandas as pd
import matplotlib.pyplot as plt

def transform_data(df):
    """
    Transforms plant order data to analyze customer ordering patterns.
    
    This function processes plant order and invoice data by filtering for 
    specific days, analyzing rolling patterns of orders for each customer, 
    and returning filtered DataFrames of invoice and order quantities.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame with columns:
        - Date: Order/invoice date
        - Monday_number: Integer indicating which Monday of the month
        - customer_id: Customer identifier
        - Invoiced_Quantity: Quantity invoiced
        - Order_Quantity: Quantity ordered
    
    Returns:
    --------
    tuple of pandas.DataFrame
        - First DataFrame: Filtered invoice quantities by date and customer
        - Second DataFrame: Corresponding order quantities for the same customers
    """
    # Convert Date to datetime and filter for the 5th Monday
    df = df.copy()  # Avoid modifying the input DataFrame
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.query('Monday_number == 5').drop(columns=['Monday_number'])
    
    # Define function to check if customer meets rolling order criteria
    def rolling_orders_6(x):
        """Check if average non-zero invoices in rolling windows exceeds threshold"""
        ts = x.set_index('Date').sort_index().Invoiced_Quantity
        # More efficient way to count non-zero values in each window
        return ts.rolling(6).apply(lambda x: (x > 0).sum()).mean() > 2
    
    # Get customers meeting the condition
    condition1 = df.groupby('customer_id').apply(rolling_orders_6)
    
    # Process Invoiced_Quantity
    invoices_grouped = df.groupby('customer_id').apply(
        lambda x: x.set_index('Date').sort_index()['Invoiced_Quantity']
    )
    invoices = invoices_grouped.reset_index().drop_duplicates().set_index(
        invoices_grouped.index.names
    ).unstack(0)
    
    # Process Order_Quantity
    orders_grouped = df.groupby('customer_id').apply(
        lambda x: x.set_index('Date').sort_index()['Order_Quantity']
    )
    orders = orders_grouped.reset_index().drop_duplicates().set_index(
        orders_grouped.index.names
    ).unstack(1)
    
    # Filter to customers meeting condition1
    selection1 = invoices['Invoiced_Quantity'].T[condition1].T
    selection1_o = orders['Order_Quantity'].T.loc[:, condition1].T
    
    # Further filter customers based on recent activity
    selection2 = selection1.T[selection1.tail(6).isna().sum() != 6].T
    selection2_o = selection1_o.loc[selection1_o.index[selection1_o.index.isin(selection2.columns)]]
    
    return selection2, selection2_o

if __name__ == "__main__":
    df = pd.read_csv('./data/raw/plant_ids.csv')
    invoices, orders = transform_data(df)
    invoices.to_csv('./data/processed/invoices.csv', index=True)
    orders.to_csv('./data/processed/orders.csv', index=True)