import numpy as np


class Conv2D(object):


    def __init__(
            self, in_channel, out_channel, kernel_size, stride, padding):
        self.W = np.random.randn(*kernel_size)
        self.b = np.random.randn(kernel_size[0], 1)
        self.kernel_size = kernel_size
        self.stride = (stride, stride) if type(stride) == int else stride
        self.padding = (padding, padding) if type(padding) == int else padding
        self.out_channel=out_channel

    def __repr__(self):
        return "{}({}, {}, {})".format(
            self.name, self.kernel_size, self.stride, self.padding
        )
    
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):


        p, s = self.padding, self.stride
        x_padded = np.pad(
            x, ((0, 0), (p[0], p[0]), (p[1], p[1])), mode='constant'
        )

        # check dimensions
        assert (x.shape[1] - self.W.shape[2] + 2 * p[0]) / s[0] + 1 > 0, \
                'Height doesn\'t work'
        assert (x.shape[2] - self.W.shape[3] + 2 * p[1]) / s[1] + 1 > 0, \
                'Width doesn\'t work'

        # TODO: Put your code below
        f = self.kernel_size[0]      #3
        t = int((x.shape[1] - f + 2 * p[0]) / s[0] + 1)   #2
        h = len(x_padded[0][0])    #6
        res = []

        for h in range(0,self.out_channel):
            a1=[[0,0,0],[0,0,0],[0,0,0]]
            for i in range(0, x.shape[0]):

                for g in range(0, t*s[0], s[0]):

                    for j in range(0, t*s[0], s[0]):

                        sum=0
                        for k in range(0, f):

                            for l in range(0, f):

                                sum+=x_padded[i][l + g][j + k]*self.W[h][i][l][k]
                        a1[g][j]+=sum
            res.append(a1)
        out=[]
        for z in range(0,self.out_channel):
            out2=[]
            for x in range(0, 4, 2):
                out1=[]
                for y in range(0, 4, 2):
                    res[z][x][y] += self.b[z]
                    out1.append(round(res[z][x][y],8))
                out2.append(out1)
            out.append(out2)
        return np.array([out])
        pass


class MaxPool2D:
    def __init__(self, kernel_size, stride, padding, name="MaxPool2D"):
        self.kernel_size = kernel_size
        self.stride = (stride, stride) if type(stride) == int else stride
        self.padding = (padding, padding) if type(padding) == int else padding
        self.name = name

    def __repr__(self):
        return "{}({}, {}, {})".format(
        self.name, self.kernel_size, self.stride, self.padding
    )

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """Compute the layer output.
        
        Arguments:
            x {[np.array]} -- the input of the layer. A 3D array of shape (
                              in_channels, in_heights, in_weights).
        Returns:
            The output of the layer. A 3D array of shape (out_channels,
                out_heights, out_weights).
        """
        p, s = self.padding, self.stride
        x_padded = np.pad(
            x, ((0, 0), (p[0], p[0]), (p[1], p[1])), mode='constant'
        )
        # check dimensions
        assert (x.shape[1] - self.kernel_size[0] + 2 * p[0]) / s[0] + 1 > 0, \
                'Height doesn\'t work'
        assert (x.shape[2] - self.kernel_size[1] + 2 * p[1]) / s[1] + 1 > 0, \
                'Width doesn\'t work'
        f = self.kernel_size[0]
        t = int((x.shape[1] - f + 2 * p[0]) / s[0] + 1)
        res = []
        for i in range(0, x.shape[0]):
            a0 = []
            for g in range(0, t*s[0],s[0]):
                a1 = []
                for j in range(0, t*s[0],s[0]):
                    a2 = []
                    for k in range(0, f):
                        a3 = []
                        for l in range(0, f):
                            a3.append(x_padded[i][l + g][j + k])
                        a2.append(a3)
                    a1.append(np.max(np.array(a2).T))
                a0.append(a1)
            res.append(a0)

        return np.array(res)

        pass


class AvgPool2D:
    def __init__(self, kernel_size, stride, padding, name="MaxPool2D"):
        self.kernel_size = kernel_size
        self.stride = (stride, stride) if type(stride) == int else stride
        self.padding = (padding, padding) if type(padding) == int else padding
        self.name = name

    def __repr__(self):
        return "{}({}, {}, {})".format(
        self.name, self.kernel_size, self.stride, self.padding
    )

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """Compute the layer output.
        
        Arguments:
            x {[np.array]} -- the input of the layer. A 3D array of shape (
                              in_channels, in_heights, in_weights).
        Returns:
            The output of the layer. A 3D array of shape (out_channels,
                out_heights, out_weights).
        """
        p, s = self.padding, self.stride
        x_padded = np.pad(
            x, ((0, 0), (p[0], p[0]), (p[1], p[1])), mode='constant'
        )

        # check dimensions
        assert (x.shape[1] - self.kernel_size[0] + 2 * p[0]) / s[0] + 1 > 0, \
                'Height doesn\'t work'
        assert (x.shape[2] - self.kernel_size[1] + 2 * p[1]) / s[1] + 1 > 0, \
                'Width doesn\'t work'


        f = self.kernel_size[0]
        t = int((x.shape[1] - f + 2 * p[0]) / s[0] + 1)

        res = []
        for i in range(0, x.shape[0]):
            a0 = []
            for g in range(0, t*s[0], s[0]):
                a1 = []
                for j in range(0, t*s[0], s[0]):
                    a2 = []
                    for k in range(0, f):
                        a3 = []
                        for l in range(0, f):
                            a3.append(x_padded[i][l + g][j + k])
                        a2.append(a3)
                    a1.append(np.mean(np.array(a2).T))
                a0.append(a1)
                # print('--------------a0')
                # print(a0)
            res.append(a0)
        # print('-------------------res')
        # print(np.array(res))
        return np.array(res)
        pass
