
class Df_Result_lib:

    @classmethod
    def prepare_df_data(cls,df):
        """this function fiil na with 0 and convert int values to string
        param: df - input pandas dataframe
        return df - prepared dataframe"""
        # replace na values by 0
        df.fillna(0)
        rows_num, col_num = df.shape
        # convert number values to string
        for col in df.columns:
            for i in range(rows_num):
                # fill not printable values ND value
                if not str(df.at[i, col]).isprintable():
                    df.at[i, col] = "ND"
                # if type of value is not string convert it to string
                elif not isinstance(df.at[i, col], str):
                    df.at[i, col] = str(df.at[i, col])
        return df

    @classmethod
    def write_result_2_file(cls,file_path, test_doc, predicted, df):
        """this function write predicted and real result to file
            params: file_path- path to file
            test_doc - array with testing values from df
            predicted - predicted values from algoritm
            df - pandas dataframe"""
        with(open(file_path, 'w')) as file:
            for feature, category in zip(test_doc, predicted):
                file.write('predicted category by nearest neighbour %r => %s\n' % (feature, category))
                file.write('category from dataframe\n')
                df_result = str(df[df['Category'] == category]) + '\n'
                file.write(df_result)
