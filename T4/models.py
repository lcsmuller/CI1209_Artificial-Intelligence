import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        return nn.DotProduct(self.w, x)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        return 1 if nn.as_scalar(self.run(x)) >= 0 else -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        batch_size = 1
        while True:
            changes = [self.w.update(nn.Constant(nn.as_scalar(y) * x.data), 1)
                        for x, y in dataset.iterate_once(batch_size)
                        if self.get_prediction(x) != nn.as_scalar(y)]
            if not changes:
                break
            
class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        self.LR = .01

        self.w1 = nn.Parameter(1, 128)
        self.b1 = nn.Parameter(1, 128)

        self.w2 = nn.Parameter(128, 64)
        self.b2 = nn.Parameter(1, 64)

        self.w3 = nn.Parameter(64, 1)
        self.b3 = nn.Parameter(1, 1)

        self.params = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        first_layer = nn.ReLU(nn.AddBias(nn.Linear(x, self.w1), self.b1))
        second_layer = nn.ReLU(nn.AddBias(nn.Linear(first_layer, self.w2), self.b2))
        return nn.AddBias(nn.Linear(second_layer, self.w3), self.b3)

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        BATCH_SIZE = 50
        TARGET_LOSS = 0.015

        l = float('inf')
        while l >= TARGET_LOSS:
            for x, y in dataset.iterate_once(BATCH_SIZE):
                l = self.get_loss(x, y)
                gradients = nn.gradients(l, self.params)
                l = nn.as_scalar(l)
                for i in range(len(self.params)): self.params[i].update(gradients[i], -self.LR)

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        self.LR = .1

        self.w1 = nn.Parameter(784, 256)
        self.b1 = nn.Parameter(1, 256)

        self.w2 = nn.Parameter(256, 128)
        self.b2 = nn.Parameter(1, 128)

        self.w3 = nn.Parameter(128, 64)
        self.b3 = nn.Parameter(1, 64)

        self.w4 = nn.Parameter(64, 10)
        self.b4 = nn.Parameter(1, 10)

        self.params = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3, self.w4, self.b4]

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        first_layer = nn.ReLU(nn.AddBias(nn.Linear(x, self.w1), self.b1))
        second_layer = nn.ReLU(nn.AddBias(nn.Linear(first_layer, self.w2), self.b2))
        third_layer = nn.ReLU(nn.AddBias(nn.Linear(second_layer, self.w3), self.b3))
        return nn.AddBias(nn.Linear(third_layer, self.w4), self.b4)

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        BATCH_SIZE = 100
        TARGET_VALID_ACC = 0.98

        l = float('inf')
        valid_acc = 0
        while valid_acc < TARGET_VALID_ACC:
            for x, y in dataset.iterate_once(BATCH_SIZE):
                l = self.get_loss(x, y)
                gradients = nn.gradients(l, self.params)
                l = nn.as_scalar(l)
                for i in range(len(self.params)): self.params[i].update(gradients[i], -self.LR)
            valid_acc = dataset.get_validation_accuracy()

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.N_CHARS = 47
        self.LANGUAGES = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        self.LR = .1

        self.init_w = nn.Parameter(self.N_CHARS, 256)
        self.init_b = nn.Parameter(1, 256)

        self.x_weight = nn.Parameter(self.N_CHARS, 256)
        self.h_weight = nn.Parameter(256, 256)

        self.b = nn.Parameter(1, 256)

        self.out_w = nn.Parameter(256, len(self.LANGUAGES))
        self.out_b = nn.Parameter(1, len(self.LANGUAGES))

        self.params = [self.init_w, self.init_b, self.x_weight, self.h_weight, self.b, self.out_w, self.out_b]

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        h = nn.ReLU(nn.AddBias(nn.Linear(xs[0], self.init_w), self.init_b))

        for char in xs[1:]:
            h = nn.ReLU(nn.AddBias(nn.Add(nn.Linear(char, self.x_weight), nn.Linear(h, self.h_weight)), self.b))

        return nn.AddBias(nn.Linear(h, self.out_w), self.out_b)

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        return nn.SoftmaxLoss(self.run(xs), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        BATCH_SIZE = 50
        TARGET_LOSS = 0.015

        l = float('inf')
        while l >= TARGET_LOSS:
            for x, y in dataset.iterate_once(BATCH_SIZE):
                l = self.get_loss(x, y)
                gradients = nn.gradients(l, self.params)
                l = nn.as_scalar(l)
                for i in range(len(self.params)): self.params[i].update(gradients[i], -self.LR)