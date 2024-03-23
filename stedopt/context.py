
import numpy

class ContextHandler:
    """
    Handles different contextual information.

    Available modes are:
    - `mean`: The average of the image
    - `quantile`: The quantiles of the image
    - `image`: The entire image
    """
    def __init__(self, ctx_info):
        """
        Initializes the class with the context information.
        
        :param ctx_info: A dictionary containing the context information
        """

        self.mode = ctx_info["mode"]
        self.use_ctx = ctx_info["use_ctx"]
        self.max = numpy.array(ctx_info["ctx_x_maxs"])
        self.min = numpy.array(ctx_info["ctx_x_mins"])

    def __call__(self, img, foreground=None):
        """
        Implements a generic `__call__` method of the class
        """
        if self.use_ctx:
            return getattr(self, f"_{self.mode}")(img, foreground)

    def _mean(self, img, foreground=None):
        """
        Calculates the average from the input array. 
        
        The average is calculated from the foreground if it is provided.

        :param img: A 2D `numpy.ndarray`
        :param foreground: (optional) A 2D `numpy.ndarray` of the foreground of the image

        :returns : The average of the array
        """
        if isinstance(foreground, type(None)):
            foreground = numpy.ones_like(img, dtype=bool)
        if not numpy.any(foreground):
            return 0.
        context = (numpy.mean(img[foreground]) - self.min) / (self.max - self.min)
        return context.item()

    def _quantile(self, img, foreground=None):
        """
        Calculates the quantiles of the foreground, with or without considering
        only the foreground.

        :param img: A 2D `numpy.ndarray`
        :param foreground: (optional) A 2D `numpy.ndarray` of the foreground of the image

        :returns : A `numpy.ndarray` of the quantiles
        """
        if isinstance(foreground, type(None)):
            foreground = numpy.ones_like(img, dtype=bool)
        if not numpy.any(foreground):
            return 0.
        context = (numpy.quantile(img[foreground], [0.05, 0.25, 0.5, 0.75, 0.95]) - self.min) / (self.max - self.min)
        return context

    def _image(self, context, *args, **kwargs):
        """
        Uses the entire image as context.

        :param img: A 2D `numpy.ndarray`

        :returns : A normalized iamge
        """
        context = context - self.min
        context = context / (self.max - self.min)
        return context
