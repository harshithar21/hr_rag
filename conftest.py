import sys, os
# Ensure the project root is on the Python path for imports in tests
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)
