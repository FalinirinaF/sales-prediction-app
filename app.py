# Features de lag si demandées
                    if use_lag_features:
                        model_data = model_data.reset_index(drop=True)
                        
                        for lag in [1, 2, 3]:
                            lag_series = model_data.groupby(['shop_id', 'item_id'])[sales_col].shift(lag)
                            model_data[f'{sales_col}_lag_{lag}'] = lag_series.values
                        
                        rolling_mean = model_data.groupby(['shop_id', 'item_id'])[sales_col].transform(lambda x: x.rolling(3, min_periods=1).mean())
                        model_data[f'{sales_col}_ma_3'] = rolling_mean
                    
                    # Features agrégées par magasin
                    if use_shop_features:
                        shop_stats = model_data.groupby(['shop_id', 'date_block_num'])[sales_col].agg(['mean', 'std']).reset_index()
                        shop_stats.columns = ['shop_id', 'date_block_num', 'shop_avg_sales', 'shop_std_sales']
                        shop_stats = shop_stats.fillna(0)
                        model_data = model_data.merge(shop_stats, on=['shop_id', 'date_block_num'], how='left')
                    
                    # Features agrégées par catégorie
                    if use_category_features:
                        category_stats = model_data.groupby(['item_category_id', 'date_block_num'])[sales_col].agg(['mean', 'std']).reset_index()
                        category_stats.columns = ['item_category_id', 'date_block_num', 'category_avg_sales', 'category_std_sales']
                        category_stats = category_stats.fillna(0)
                        model_data = model_data.merge(category_stats, on=['item_category_id', 'date_block_num'], how='left')
