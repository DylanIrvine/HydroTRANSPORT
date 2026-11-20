# the xlsx download

import io

import pandas as pd


def build(sheets):
    # sheets is a dict of sheet name to DataFrame, written in order. Returns the file as bytes
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        bold = writer.book.add_format({'bold': True, 'text_wrap': True, 'valign': 'top'})
        for name, table in sheets.items():
            table.to_excel(writer, sheet_name=name[:31], index=False)
            sheet = writer.sheets[name[:31]]
            for column, title in enumerate(table.columns):
                sheet.write(0, column, str(title), bold)
                width = max(10, min(45, len(str(title)) + 2))
                if column < 2 and table.dtypes.iloc[column] == object:
                    width = max(width, min(70, int(table.iloc[:, column].astype(str).str.len().max()) + 2)) if len(table) else width
                sheet.set_column(column, column, width)
            sheet.freeze_panes(1, 0)
    return buffer.getvalue()
