class Foo:
	def __init__(self,x):
		self.x = x
	def func(self,a):
		return self.x+a
	def gunc(self,meth,a):
		return getattr(self,meth)(a)