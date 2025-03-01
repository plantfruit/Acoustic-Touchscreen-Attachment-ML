from graphviz import Digraph

dot = Digraph(comment='Data Processing Pipeline')

dot.node('A', 'Data Collection')
dot.node('B', 'Data Cleaning')
dot.node('C', 'Feature Extraction')
dot.node('D', 'Model Training')
dot.node('E', 'Evaluation')
dot.node('F', 'Deployment')

dot.edges(['AB', 'BC', 'CD', 'DE', 'EF'])

dot.render('pipeline', view=True)  # Save and view the diagram
