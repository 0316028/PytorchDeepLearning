
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from caffe2.python import brew
'''
Utility for creating ResNets
See "Deep Residual Learning for Image Recognition" by He, Zhang et. al. 2015
'''


class ResNetBuilder():
    '''
    Helper class for constructing residual blocks.
    '''

    def __init__(self, model, prev_blob, no_bias, is_test, spatial_bn_mom=0.9):
        self.model = model
        self.comp_count = 0
        self.comp_idx = 0
        self.prev_blob = prev_blob
        self.is_test = is_test
        self.spatial_bn_mom = spatial_bn_mom
        self.no_bias = 1 if no_bias else 0

    def add_conv(self, in_filters, out_filters, kernel, stride=1, pad=0):
        self.comp_idx += 1
        self.prev_blob = brew.conv(
            self.model,
            self.prev_blob,
            'comp_%d_conv_%d' % (self.comp_count, self.comp_idx),
            in_filters,
            out_filters,
            weight_init=("MSRAFill", {}),
            kernel=kernel,
            stride=stride,
            pad=pad,
            no_bias=self.no_bias,
        )
        return self.prev_blob

    def add_relu(self):
        self.prev_blob = brew.relu(
            self.model,
            self.prev_blob,
            self.prev_blob,  # in-place
        )
        return self.prev_blob

    def add_spatial_bn(self, num_filters):
        self.prev_blob = brew.spatial_bn(
            self.model,
            self.prev_blob,
            'comp_%d_spatbn_%d' % (self.comp_count, self.comp_idx),
            num_filters,
            epsilon=1e-3,
            momentum=self.spatial_bn_mom,
            is_test=self.is_test,
        )
        return self.prev_blob

    '''
    Add a "bottleneck" component as decribed in He et. al. Figure 3 (right)
    '''

    def add_bottleneck(
        self,
        input_filters,   # num of feature maps from preceding layer
        base_filters,    # num of filters internally in the component
        output_filters,  # num of feature maps to output
        down_sampling=False,
        spatial_batch_norm=True,
    ):
        self.comp_idx = 0
        shortcut_blob = self.prev_blob

        # 1x1
        self.add_conv(
            input_filters,
            base_filters,
            kernel=1,
            stride=1
        )

        if spatial_batch_norm:
            self.add_spatial_bn(base_filters)

        self.add_relu()

        # 3x3 (note the pad, required for keeping dimensions)
        self.add_conv(
            base_filters,
            base_filters,
            kernel=3,
            stride=(1 if down_sampling is False else 2),
            pad=1
        )

        if spatial_batch_norm:
            self.add_spatial_bn(base_filters)
        self.add_relu()

        # 1x1
        last_conv = self.add_conv(base_filters, output_filters, kernel=1)
        if spatial_batch_norm:
            last_conv = self.add_spatial_bn(output_filters)

        # Summation with input signal (shortcut)
        # If we need to increase dimensions (feature maps), need to
        # do a projection for the short cut
        if (output_filters > input_filters):
            shortcut_blob = brew.conv(
                self.model,
                shortcut_blob,
                'shortcut_projection_%d' % self.comp_count,
                input_filters,
                output_filters,
                weight_init=("MSRAFill", {}),
                kernel=1,
                stride=(1 if down_sampling is False else 2),
                no_bias=self.no_bias,
            )
            if spatial_batch_norm:
                shortcut_blob = brew.spatial_bn(
                    self.model,
                    shortcut_blob,
                    'shortcut_projection_%d_spatbn' % self.comp_count,
                    output_filters,
                    epsilon=1e-3,
                    momentum=self.spatial_bn_mom,
                    is_test=self.is_test,
                )

        self.prev_blob = brew.sum(
            self.model, [shortcut_blob, last_conv],
            'comp_%d_sum_%d' % (self.comp_count, self.comp_idx)
        )
        self.comp_idx += 1
        self.add_relu()

        # Keep track of number of high level components if this ResNetBuilder
        self.comp_count += 1

    def add_simple_block(
        self,
        input_filters,
        num_filters,
        down_sampling=False,
        spatial_batch_norm=True
    ):
        self.comp_idx = 0
        shortcut_blob = self.prev_blob

        # 3x3
        self.add_conv(
            input_filters,
            num_filters,
            kernel=3,
            stride=(1 if down_sampling is False else 2),
            pad=1
        )

        if spatial_batch_norm:
            self.add_spatial_bn(num_filters)
        self.add_relu()

        last_conv = self.add_conv(num_filters, num_filters, kernel=3, pad=1)
        if spatial_batch_norm:
            last_conv = self.add_spatial_bn(num_filters)

        # Increase of dimensions, need a projection for the shortcut
        if (num_filters != input_filters):
            shortcut_blob = brew.conv(
                self.model,
                shortcut_blob,
                'shortcut_projection_%d' % self.comp_count,
                input_filters,
                num_filters,
                weight_init=("MSRAFill", {}),
                kernel=1,
                stride=(1 if down_sampling is False else 2),
                no_bias=self.no_bias,
            )
            if spatial_batch_norm:
                shortcut_blob = brew.spatial_bn(
                    self.model,
                    shortcut_blob,
                    'shortcut_projection_%d_spatbn' % self.comp_count,
                    num_filters,
                    epsilon=1e-3,
                    is_test=self.is_test,
                )

        self.prev_blob = brew.sum(
            self.model, [shortcut_blob, last_conv],
            'comp_%d_sum_%d' % (self.comp_count, self.comp_idx)
        )
        self.comp_idx += 1
        self.add_relu()

        # Keep track of number of high level components if this ResNetBuilder
        self.comp_count += 1
def create_resnet50(
    model,
    data,
    num_input_channels,
    num_labels,
    label=None,
    is_test=False,
    no_loss=False,
    no_bias=0,
    conv1_kernel=7,
    conv1_stride=2,
    final_avg_kernel=7,
):
    # conv1 + maxpool
    brew.conv(
        model,
        data,
        'conv1',
        num_input_channels,
        64,
        weight_init=("MSRAFill", {}),
        kernel=conv1_kernel,
        stride=conv1_stride,
        pad=3,
        no_bias=no_bias
    )

    brew.spatial_bn(
        model,
        'conv1',
        'conv1_spatbn_relu',
        64,
        epsilon=1e-3,
        momentum=0.1,
        is_test=is_test
    )
    brew.relu(model, 'conv1_spatbn_relu', 'conv1_spatbn_relu')
    brew.max_pool(model, 'conv1_spatbn_relu', 'pool1', kernel=3, stride=2)

    # Residual blocks...
    builder = ResNetBuilder(model, 'pool1', no_bias=no_bias,
                            is_test=is_test, spatial_bn_mom=0.1)

    # conv2_x (ref Table 1 in He et al. (2015))
    builder.add_bottleneck(64, 64, 256)
    builder.add_bottleneck(256, 64, 256)
    builder.add_bottleneck(256, 64, 256)

    # conv3_x
    builder.add_bottleneck(256, 128, 512, down_sampling=True)
    for _ in range(1, 4):
        builder.add_bottleneck(512, 128, 512)

    # conv4_x
    builder.add_bottleneck(512, 256, 1024, down_sampling=True)
    for _ in range(1, 6):
        builder.add_bottleneck(1024, 256, 1024)

    # conv5_x
    builder.add_bottleneck(1024, 512, 2048, down_sampling=True)
    builder.add_bottleneck(2048, 512, 2048)
    builder.add_bottleneck(2048, 512, 2048)

    # Final layers
    final_avg = brew.average_pool(
        model,
        builder.prev_blob,
        'final_avg',
        kernel=final_avg_kernel,
        stride=1,
    )

    # Final dimension of the "image" is reduced to 7x7
    last_out = brew.fc(
        model, final_avg, 'last_out_L{}'.format(num_labels), 2048, num_labels
    )

    if no_loss:
        return last_out

    # If we create model for training, use softmax-with-loss
    if (label is not None):
        (softmax, loss) = model.SoftmaxWithLoss(
            [last_out, label],
            ["softmax", "loss"],
        )

        return (softmax, loss)
    else:
        # For inference, we just return softmax
        return brew.softmax(model, last_out, "softmax")
def create_mnist( 
    model, data, 
    num_input_channels, 
    num_labels, 
    label=None, 
    is_test=False, 
    no_loss=False, 
    no_bias=0, 
    conv1_kernel=7, 
    conv1_stride=2, 
    final_avg_kernel=7, ): 
    # Image size: 28 x 28 -> 24 x 24 
    conv1 = brew.conv(model,data, 'conv1', dim_in=1, dim_out=20, kernel=5) 
    # Image size: 24 x 24 -> 12 x 12 
    pool1 = brew.max_pool(model,conv1, 'pool1', kernel=2, stride=2) 
    # Image size: 12 x 12 -> 8 x 8 
    conv2 = brew.conv(model,pool1, 'conv2', dim_in=20, dim_out=50, kernel=5) 
    # Image size: 8 x 8 -> 4 x 4 
    pool2 = brew.max_pool(model,conv2, 'pool2', kernel=2, stride=2) 
    # 50 * 4 * 4 stands for dim_out from previous layer multiplied by the image size 
    fc3 = brew.fc(model,pool2, 'fc3', dim_in=50 * 4* 4, dim_out=500)
    fc3 = brew.relu(model,fc3, fc3) 
    
    pred = brew.fc(model,fc3, 'pred', 500, 10) 
    
    if no_loss:
        print("no_loss")
        return pred

    # If we create model for training, use softmax-with-loss
    if (label is not None):
        softmax = brew.softmax(model, pred, 'softmax')
        xent = model.LabelCrossEntropy([softmax, label], 'xent')
        # compute the expected loss
        loss = model.AveragedLoss(xent, "loss")
        '''(softmax, loss) = model.SoftmaxWithLoss(
            [pred, label],
            ["softmax", "loss"],
        )'''

        return (softmax, loss)
    else:
        # For inference, we just return softmax
        return brew.softmax(model, pred, "softmax")
    '''if no_loss:
        print("no_loss")
        return pred

    # If we create model for training, use softmax-with-loss
    if (label is not None):
        (softmax, loss) = model.SoftmaxWithLoss(
            [pred, label],
            ["softmax", "loss"],
        )

        return (softmax, loss)
    else:
        # For inference, we just return softmax
        return brew.softmax(model, pred, "softmax")'''
def Create_Cnn(
    model,
    data,
    num_input_channels,
    num_labels,
    label=None,
    is_test=False,
    no_loss=False,
    no_bias=0,
    conv1_kernel=7,
    conv1_stride=2,
    final_avg_kernel=7,
):
    conv1 = brew.conv(model, data, 'conv1', dim_in=num_input_channels, dim_out=32, kernel=5)
    # Image size: 24 x 24 -> 12 x 12
    pool1 = brew.max_pool(model, conv1, 'pool1', kernel=3, stride=2)
    # Image size: 12 x 12 -> 8 x 8
    conv2 = brew.conv(model, pool1, 'conv2', dim_in=32, dim_out=48, kernel=3)
    # Image size: 8 x 8 -> 4 x 4
    pool2 = brew.max_pool(model, conv2, 'pool2', kernel=3, stride=2)
    # 50 * 4 * 4 stands for dim_out from previous layer multiplied by the image size
    conv3 = brew.conv(model, pool2, 'conv3', dim_in=48, dim_out=64, kernel=3)
    fc3 = brew.fc(model, conv3, 'fc3', dim_in=64 * 2 * 2, dim_out=500)
    fc3 = brew.relu(model, fc3, fc3)
    dropout1 = brew.dropout(model, fc3, 'dropout1', ratio=0.5, is_test=0)
    
    pred = brew.fc(model, dropout1, 'pred', 500, num_labels)
    # 50 * 4 * 4 stands for dim_out from previous layer multiplied by the image size
    #pred1 = brew.fc(model, data, 'pred1', num_input_channels * 28 * 28, 500)
    #pred = brew.fc(model, pred1, 'pred', 500, num_labels)
    #pred = brew.fc(model, fc3, 'pred', 500, num_labels)
    #pred = brew.fc(model, data, 'pred', num_input_channels*227*227, num_labels)
    
    if no_loss:
        print("no_loss")
        return pred

    # If we create model for training, use softmax-with-loss
    if (label is not None):
        '''softmax = brew.softmax(model, pred, 'softmax')
        xent = model.LabelCrossEntropy([softmax, label], 'xent')
        # compute the expected loss
        loss = model.AveragedLoss(xent, "loss")'''
        (softmax, loss) = model.SoftmaxWithLoss(
            [pred, label],
            ["softmax", "loss"],
        )

        return (softmax, loss)
    else:
        # For inference, we just return softmax
        return brew.softmax(model, pred, "softmax")
