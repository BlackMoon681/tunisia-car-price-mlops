from graphviz import Digraph

dot = Digraph(
    comment='Star Schema with Fact Centered',
    graph_attr={
        'rankdir': 'LR',
        'splines': 'ortho',
        'ratio': 'compress',  # tightly fit graph to content
        'dpi': '150'
    }
)

table_style = {'shape': 'plaintext'}

fact_table_html = '''<
<TABLE BORDER="1" CELLBORDER="1" CELLSPACING="0" CELLPADDING="1">
    <TR><TD BGCOLOR="lightgrey"><B><FONT FACE="Helvetica" POINT-SIZE="10">FactsTable</FONT></B></TD></TR>
    <TR><TD><FONT FACE="Helvetica" POINT-SIZE="9">ID_Dossier</FONT></TD></TR>
    <TR><TD><FONT FACE="Helvetica" POINT-SIZE="9">ID_Clients</FONT></TD></TR>
    <TR><TD><FONT FACE="Helvetica" POINT-SIZE="9">ID_Cars</FONT></TD></TR>
    <TR><TD><FONT FACE="Helvetica" POINT-SIZE="9">ID_Evaluation</FONT></TD></TR>
    <TR><TD><FONT FACE="Helvetica" POINT-SIZE="9">ID_Staff</FONT></TD></TR>
    <TR><TD><FONT FACE="Helvetica" POINT-SIZE="9">Performance_Act</FONT></TD></TR>
    <TR><TD><FONT FACE="Helvetica" POINT-SIZE="9">Rapidité_service</FONT></TD></TR>
    <TR><TD><FONT FACE="Helvetica" POINT-SIZE="9">Rendement</FONT></TD></TR>
</TABLE>>'''

dot.node('FactsTable', fact_table_html, style='filled', fillcolor='lightgrey', **table_style)

dim_tables = {
    'Clients': ['ID_Clients', 'Nom', 'Prenom', 'Tel', 'Email'],
    'Staff': ['ID_Staff', 'Nom', 'Prenom', 'Email', 'Login', 'Access_Code', 'Agence', 'Date_Creation'],
    'Cars': ['ID_Cars', 'Date_Creation', 'Marque', 'Modèle', 'Finition', 'Kilométrage', 'Carburant', 'Type', 'Numéro_Châssis'],
    'Evaluation': ['ID_Evaluation', 'Date_Creation', 'Finition', 'Prix_Marche_Mean', 'Prix_Reprise', 'Total_Reparation'],
    'DIMDATE': ['ID_Date', 'MyDate', 'Day', 'Month', 'Year', 'DateString']
}

dim_names = list(dim_tables.keys())
mid_index = len(dim_names) // 2
left_dims = dim_names[:mid_index]
right_dims = dim_names[mid_index:]

def create_dim_node(name, fields):
    rows = ''.join(f'<TR><TD><FONT FACE="Helvetica" POINT-SIZE="9">{field}</FONT></TD></TR>' for field in fields)
    label = f'''<
    <TABLE BORDER="1" CELLBORDER="1" CELLSPACING="0" CELLPADDING="1">
        <TR><TD BGCOLOR="lightblue"><B><FONT FACE="Helvetica" POINT-SIZE="10">{name}</FONT></B></TD></TR>
        {rows}
    </TABLE>>'''
    dot.node(name, label, **table_style)

for dim in dim_names:
    create_dim_node(dim, dim_tables[dim])

# Invisible anchors for layout control
dot.node('left_anchor', label='', width='0', style='invis')
dot.node('right_anchor', label='', width='0', style='invis')

with dot.subgraph() as left_rank:
    left_rank.attr(rank='same')
    left_rank.node('left_anchor')
    for dim in left_dims:
        left_rank.node(dim)

with dot.subgraph() as right_rank:
    right_rank.attr(rank='same')
    right_rank.node('right_anchor')
    for dim in right_dims:
        right_rank.node(dim)

with dot.subgraph() as fact_rank:
    fact_rank.attr(rank='same')
    fact_rank.node('FactsTable')

# Force horizontal ordering with invisible edges
dot.edge('left_anchor', 'FactsTable', style='invis', weight='10')
dot.edge('FactsTable', 'right_anchor', style='invis', weight='10')

# Connect dims to fact table with visible edges but don't affect layout
for dim in left_dims + right_dims:
    dot.edge(dim, 'FactsTable', color='blue', constraint='false')

dot.render('star_schema_centered_fact', format='png', view=True)
