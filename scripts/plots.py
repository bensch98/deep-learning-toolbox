import sys
import os
import matplotlib.pyplot as plt
import numpy as np


class Plot:

  def __init__(self, x, func, title):
    """ Init activation function.
    :param x: Input x of activation function f(x).
    :param func: Activation function f(x).
    """
    self.x = x
    self.func = func
    self.func = [(i, None) if not type(i) is tuple else i for i in self.func]
    self.title = title

    # save figure and plot globally
    self.fig = None
    self.plot = None

  def create_plot(self):
    """ Plots the activation function.
    :param title: Title of the plot.
    :param save: If yes, saves the plot, instead of showing it.
    """

    self.fig = plt.figure(figsize=(10,5))
    for func, label in self.func:
      plt.plot(self.x, func, label=label)
      if label is not None:
        plt.legend(loc='upper right')
        plt.ylim(-1.5, 2.0)
    plt.axis('tight')
    plt.title(self.title)
    self.plot = plt

    return self


class Distribution:

  def __init__(self):
    pass
  
  def gaussian(self, x, mu, sig):
    return 1./(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)


def save_plots(plts):
  # save activation functions
  for p in plts:
    filename = p.title.lower().replace(' ', '_')
    path = os.path.relpath(f'./img/{filename}')
    # defaults to png image
    p.fig.savefig(path, bbox_inches='tight', dpi=150)

def show_plots(plts):
  # show activation functions
  for p in plts:
    p.plot.show()


if __name__ == '__main__':
  act_funcs = []

  # same input x for all activation functions
  x = np.linspace(-10, 10)

  # binary step activation function
  y = np.heaviside(x, 1)
  binary_step = Plot(x, [y], 'Binary Step').create_plot()
  act_funcs.append(binary_step)
  
  # linear activation fucntion
  y = x
  linear = Plot(x, [y], 'Linear').create_plot()
  act_funcs.append(linear)
  
  # sigmoid activation function
  y = 1/(1+np.exp(-x))
  sigmoid = Plot(x, [y], 'Sigmoid').create_plot() 
  act_funcs.append(sigmoid)

  # tanh activation function
  y = np.tanh(x)
  tanh = Plot(x, [y], 'Tanh').create_plot()
  act_funcs.append(tanh)
  
  # rectified linear unit (ReLU)
  y = np.maximum(0, x)
  relu = Plot(x, [y], 'ReLU').create_plot()
  act_funcs.append(relu)

  # softmax activation function
  y = np.exp(x)/np.sum(np.exp(x), axis=0)
  softmax = Plot(x, [y], 'Softmax').create_plot()
  act_funcs.append(softmax)

  # distributions
  x = np.linspace(-5, 5)
  normal_dists = []
  normal_dists.append((Distribution().gaussian(x, 0, 1), r'$\mu=0, \sigma^2=1$'))
  normal_dists.append((Distribution().gaussian(x, 0, 0.8), r'$\mu=0, \sigma^2=0.8$'))
  normal_dists.append((Distribution().gaussian(x, 0, 5), r'$\mu=0, \sigma^2=5$'))
  normal_dists.append((Distribution().gaussian(x, -2, 0.5), r'$\mu=-2, \sigma^2=0.5$'))
  normal_dist = Plot(x, normal_dists, 'Gaussian Distribution').create_plot()
  act_funcs.append(normal_dist)

  
  arg1 = sys.argv[1]
  if arg1 == 'show':
    show_plots(act_funcs)
  elif arg1 == 'save':
    save_plots(act_funcs)
  else:
    print('First argument must either be "show" or "save"')
