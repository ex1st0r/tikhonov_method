#! /usr/bin/env python
# -*- coding: UTF-8 -*-
from __future__ import division #Убираем проблему целочисленного деления в Python 2.7
from numpy import dot, random, sqrt, array, linspace, exp, eye,vstack, set_printoptions, linalg, fabs, zeros
from matplotlib import pyplot as plt
import time

time_l = time.time() #Счетчик времени выполнения программы

class TikhonovMethod:
	
	def solve(self, a = 0, b = 1, c = 0, d = 1, n = 41, m = 41, delta = 10**-8, h = 10**-10):
		self.n = n
		self.m = m
		self.hS = (b - a) / (n - 1)
		self.hX = (d - c) / (m - 1)
		self.h = h
		s = linspace(a, b, n)
		x = linspace(c, d, m)

		K = array(self.K(x,s))
		u = dot(K,self.z(s))*self.hS

		#Начальное приближение
		z = random.random_sample((n,))*-10
		alpha = 0.01

		z_min, alpha = self.chords_method(K,z,u,alpha, delta, h)
		print('z_min=', z_min,'alpha=', alpha)

		#Рисуем график
		plt.plot(s,self.z(s), s, z_min, 'ro')
		plt.title(u'Минимизация функционала Тихонова', family='verdana')
		plt.xlabel('s')
		plt.ylabel('z(s)')
		plt.legend((r'$z_0$', r"$z_\eta^\alpha$"), loc=(0.85,0.85))
		plt.grid()
		plt.text (-0.1, -0.1, u'Время выполнения: %.3f секунд' % (time.time() - time_l), family='verdana', size=8)
		plt.show()

	def kernel_func(self, x, s):
			return 1/(1+100*(x-s)**2)

	def K(self,x,s):#Полиморфная функция
		if (type(x) == type(s) == type(array([]))):
			K_x_s = []
			for x_i in x:
				a = []
				for s_j in s:
					a.append(self.kernel_func(x_i,s_j))
				a[0] = a[0] / 2
				a[-1] = a[-1] / 2
				K_x_s.append(a)
			return K_x_s

		return self.kernel_func(x,s)

	def z_func(self, s):
		return exp(-(s-0.5)**2/0.06)
		#return 4*s*(1-s)
		#return (1-s**2)

	def z(self, s):#Полиморфная функция
		if (type(s) == type(array([]))):
			res = []
			for s_i in s:
				res.append(self.z_func(s_i))
			return res

		return self.z_func(s)

	def chords_method(self, K,z,u,alpha, delta,h):#Метод хорд
		f = self.B_F(K, u)
		B = self.B_alpha(K,z,u,alpha)
		z = self.conjugate_gradient(B, z, f)
		F0 = self.genDiscrepancy(K,z,u,delta,h)
		while (F0 < -h):
			alpha *= 2
			B = self.B_alpha(K,z,u,alpha)
			z = self.conjugate_gradient(B, z, f)
			F0 = self.genDiscrepancy(K,z,u,delta,h)
			print('po < eps')
		x0 = 1/alpha
		alpha *= 2
		x=1/(alpha)
		po = F0
		for iter in xrange(100):
			if (po < -h):
				F = po
				w_iters = 100
				while ((po < -h) & bool(w_iters)):
					w_iters -= 1
					print('w_iters=', w_iters)
					y = x0 + F*(x0-x)/(F0-F)
					alpha = 1/y
					B = self.B_alpha(K,z,u,alpha)
					z = self.conjugate_gradient(B, z, f)
					po = self.genDiscrepancy(K,z,u,delta,h)
					if (po < -h):
						x = y
						F = po
						continue
					x0 = y
					F0 = po
				print('EXIT BECAUSE END MOD_METH_CHORDS on iter=', iter, 'w_iters=', w_iters)
				return z, alpha

			if (fabs(po) < h):
				print('|po| < eps; Alfa was founded by Newton\'s method! on iter=', iter)
				break
			B = self.B_alpha(K,z,u,alpha)
			z = self.conjugate_gradient(B, z, f)
			po = self.genDiscrepancy(K,z,u,delta,h)
			y = x0 - F0/(F0-po)*(x0-x)
			x0 = x
			x=y
			F0 = po
			alpha = 1/x
		return z, alpha

	def genDiscrepancy(self, A, z, u, delta, h):#Обобщенная невязка
		nev = (((dot(A, z) * self.hS - u) **2) * self.hX).sum()
		z = array(z)
		z_1 = (z*z).sum() * self.hS
		z_2 = 0
		for i in xrange(1,z.size):
			z_2 += (z[i]-z[i-1])**2
		z_2 /= self.hS
		norm_z_V2 = sqrt(z_1 + z_2)
		#print('nev=', nev)
		p = nev - (delta + h*norm_z_V2)**2
		return p

	def conjugate_gradient(self,A, x0, b):#Метод сопр. градиентов
		x = x0
		r0 = b - dot(A, x)
		p = r0

		for i in xrange(100):
			a = dot(r0.T, r0) / dot(dot(p.T, A), p)
			x = x + p*a
			ri = r0 - dot(A*a, p)
			if linalg.norm(ri) < self.h:
				return x
			b = dot(ri.T, ri) / dot(r0.T, r0)
			p = ri + b * p
			r0 = ri
		return x

	def B_alpha(self, K, z, u, alpha):#Левая часть уравнения Эйлера
		B = []
		for j in xrange(self.n):
			b_j = []
			for k in xrange(self.n):
				value = self.hX*(K[:,k]*K[:,j]).sum()
				b_j.append(value)
			B.append(b_j)
		C = eye(self.n)*(1+2/(self.hS**2))
		C[0,0] = C[0,0]/2 + 0.5
		C[-1,-1] = C[0,0]
		
		C += eye(self.n, k=1)*(-1/(self.hS**2))
		C += eye(self.n, k=-1)*(-1/(self.hS**2))
		B = array(B)
		B = self.hS*(B+alpha*C)
		return B

	def B_F(self, K, u):#Правая часть уравнения Эйлера
		f = []
		K = array(K)
		u = array(u)
		for j in xrange(self.n):
			f.append((K[:,j]*u).sum() * self.hX)
		return array(f)

a = TikhonovMethod()
a.kernel_func = lambda x,s: 1/(1+100*(x-s)**2) #Ядро
a.z_func = lambda s: exp(-(s-0.5)**2/0.06) #Точное решение z
a.solve(a = 0, b = 1, c = 0, d = 1, n = 41, m = 41, delta = 10**-8, h = 10**-10) #Запуск процесса