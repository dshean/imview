#! /usr/bin/env python

#David Shean
#dshean@gmail.com

#Load custom color ramp from cpt
#gmtColormap function from: http://wiki.scipy.org/Cookbook/Matplotlib/Loading_a_colormap_dynamically

#To use:
#import gmtColormap
#cpt_rainbow = gmtColormap.get_rainbow()

def get_rainbow(rev=False):
    import os
    import matplotlib.colors 
    #Check for bundled cpt file in script directory
    fn_path = os.path.dirname(os.path.realpath(__file__))
    fn = os.path.join(fn_path,'rainbow.cpt')
    d = gmtColormap(fn, reverse=rev)
    name = 'cpt_rainbow'
    if rev:
        name = 'cpt_rainbow_r'
    cmap = matplotlib.colors.LinearSegmentedColormap(name, d)
    return cmap

def gmtColormap(fileName,GMTPath = None, reverse=False):
      import colorsys
      import numpy as np 
      #if type(GMTPath) == type(None):
      #    filePath = "/usr/local/cmaps/"+ fileName+".cpt"
      #else:
      #    filePath = GMTPath+"/"+ fileName +".cpt"
      filePath = fileName
      try:
          f = open(filePath)
      except:
          print("file ",filePath, "not found")
          return None

      lines = f.readlines()
      f.close()
      
      x = []
      r = []
      g = []
      b = []
      colorModel = "RGB"
      for l in lines:
          ls = l.split()
          if l[0] == "#":
             if ls[-1] == "HSV":
                 colorModel = "HSV"
                 continue
             else:
                 continue
          if ls[0] == "B" or ls[0] == "F" or ls[0] == "N":
             pass
          else:
              x.append(float(ls[0]))
              r.append(float(ls[1]))
              g.append(float(ls[2]))
              b.append(float(ls[3]))
              xtemp = float(ls[4])
              rtemp = float(ls[5])
              gtemp = float(ls[6])
              btemp = float(ls[7])

      x.append(xtemp)
      r.append(rtemp)
      g.append(gtemp)
      b.append(btemp)

      nTable = len(r)
      x = np.array(x, float)
      r = np.array(r, float)
      g = np.array(g, float)
      b = np.array(b, float)

      if reverse:
        r = r[::-1]
        g = g[::-1]
        b = b[::-1]

      if colorModel == "HSV":
         for i in range(r.shape[0]):
             rr,gg,bb = colorsys.hsv_to_rgb(r[i]/360.,g[i],b[i])
             r[i] = rr ; g[i] = gg ; b[i] = bb
      if colorModel == "HSV":
         for i in range(r.shape[0]):
             rr,gg,bb = colorsys.hsv_to_rgb(r[i]/360.,g[i],b[i])
             r[i] = rr ; g[i] = gg ; b[i] = bb
      if colorModel == "RGB":
          r = r/255.
          g = g/255.
          b = b/255.
      xNorm = (x - x[0])/(x[-1] - x[0])

      red = []
      blue = []
      green = []
      for i in range(len(x)):
          red.append([xNorm[i],r[i],r[i]])
          green.append([xNorm[i],g[i],g[i]])
          blue.append([xNorm[i],b[i],b[i]])
      colorDict = {"red":red, "green":green, "blue":blue}
      return (colorDict)
