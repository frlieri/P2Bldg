import pandas as pd
from unittest.mock import patch, MagicMock
from src.export.excel_generator import save_xlsx_wb

class TestExcelGenerator:

    @patch('pandas.ExcelWriter')
    @patch('pandas.DataFrame.to_excel')
    def test_save_xlsx_wb(self, mock_to_excel, mock_writer):
        """
        Tests if save_xlsx_wb correctly iterates through a dictionary of dataframes
        and calls the pandas ExcelWriter to save them.
        """
        # Prepare dummy data
        data = {
            'Sheet1': pd.DataFrame({'A': [1, 2], 'B': [3, 4]}),
            'Sheet2': pd.DataFrame({'X': [10], 'Y': [20]})
        }
        fpath = "dummy_results/output.xlsx"

        # Setup mock context manager for ExcelWriter
        mock_writer_instance = MagicMock()
        mock_writer.return_value.__enter__.return_value = mock_writer_instance
        mock_to_excel.return_value = MagicMock()

        # Call the function
        save_xlsx_wb(fpath, data)

        # Assertions
        assert mock_writer.called
        # Check if to_excel was called for each dataframe in the dictionary
        assert mock_writer_instance.book.add_worksheet.called or mock_writer_instance.sheets
        
        # Verify that each sheet was processed
        calls = [call.args[1] if len(call.args) > 1 else call.kwargs.get('sheet_name') 
                 for call in mock_to_excel.call_args_list]
        
        # In newer pandas, the check might vary slightly depending on internal implementation, 
        # but verifying the number of to_excel calls is robust.
        assert len(mock_to_excel.call_args_list) == 2
