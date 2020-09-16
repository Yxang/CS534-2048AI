import importlib.machinery
loader = importlib.machinery.SourceFileLoader('g', "2048-api/game2048/game.py")
g = loader.load_module()

game = g.Game()