package dasom;

import java.util.*;

class DenoisingAutoencoderSelfOrganizingMap {
    
    int L;
    int M;
    double W[][];
    double b[];
    double b_hat[];
    boolean bias;
    boolean sigmOrTanh;
    boolean linearDecoder;
    int neuronsPerColumn;
    int neuronsPerRow;
    int E;
    double U[][];
    double neuronPosition[][];
   
    /*
    Constructor for a hexagonal DASOM lattice using the appropriate number of
    per column and per row neurons. For instance a 4x7 DASOM is represented by an
    array with 4 rows and 7 columns, in this case the constructor call should be
    DenoisingAutoencoderSelfOrganzingMap(4, 7). Positions are stored starting
    from the lower left neuron and proceeding in a left-to-right down-to-up
    fashion until the last neuron which stored in the upper right. Details of the
    employed architecture, namely bias factor inclusion, squashing function type
    and the option of a linear function in the decoding stage, are defined. In
    addition, the number of each sample's features L (i.e. input dimensionality)
    and the number of hidden layer representations M is selected. It should be
    noted that all parameter matrices are initialized according to the commonly
    used N(0, 0.01^2) normal distribution. For the U matrix this decision is
    more justified in the hyperbolic tangent case because in allows both positive
    and negative values.
    */
    DenoisingAutoencoderSelfOrganizingMap(int L, int M, boolean bias,
                                          boolean sigmOrTanh, boolean linearDecoder,
                                          int neuronsPerColumn, int neuronsPerRow) {
        this.L = L;
        this.M = M;
        W = new double[M][L];
        b = new double[M];
        b_hat = new double[L];
        this.bias = bias;
        this.sigmOrTanh = sigmOrTanh;
        this.linearDecoder = linearDecoder;
        this.neuronsPerColumn = neuronsPerColumn;
        this.neuronsPerRow = neuronsPerRow;
        E = neuronsPerColumn * neuronsPerRow;
        U = new double[M][E];
        neuronPosition = new double[E][2];

        //initialization
        Random variable = new Random();
        double standardDeviation = 0.01;
        if (bias) {
            for (int i = 0; i < M; i++)
                b[i] = variable.nextGaussian() * standardDeviation;
            for (int j = 0; j < L; j++)
                b_hat[j] = variable.nextGaussian() * standardDeviation;                
        }
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < L; j++)
                W[i][j] = variable.nextGaussian() * standardDeviation;
            for (int e = 0; e < E; e++)
                U[i][e] = variable.nextGaussian() * standardDeviation;
        }

        double stepX = 1.0;
        double offsetX = 0.5;
        double stepY = Math.sin(Math.toRadians(60));
        double posX;
        double posY = 0.0;
        for (int dimY = 0; dimY<neuronsPerColumn; dimY++) {
            if (dimY % 2 == 0) 
                posX = 0.0;
            else 
                posX = offsetX;
            for (int dimX = 0; dimX<neuronsPerRow; dimX++) {
                neuronPosition[dimY*neuronsPerRow+dimX][0] = posX;
                neuronPosition[dimY*neuronsPerRow+dimX][1] = posY;
                posX += stepX;
            }
            posY += stepY;
        }
    }
    
    /*
    Constructor for a orthogonal/hexagonal DASOM lattice using the appropriate
    number of per column and per row neurons. For instance 8x5 DASOM is represented
    by an array with 8 rows and 5 columns, in this case the constructor call should
    be DenoisingAutoencoderSelfOrganzingMap(8, 5). Positions are stored starting
    from the lower left neuron and proceeding in a left-to-right down-to-up
    fashion until the last neuron which stored in the upper right. Details of the
    employed architecture, namely bias factor inclusion, squashing function type
    and the option of a linear function in the decoding stage, are defined. In
    addition, the number of each sample's features L (i.e. input dimensionality)
    and the number of hidden layer representations M is selected. It should be
    noted that all parameter matrices are initialized according to the commonly
    used N(0, 0.01^2) normal distribution. For the U matrix this decision is
    more justified in the hyperbolic tangent case because in allows both positive
    and negative values.
    */
    DenoisingAutoencoderSelfOrganizingMap(int L, int M, boolean bias,
            boolean sigmOrTanh, boolean linearDecoder, int neuronsPerColumn,
            int neuronsPerRow, boolean orthOrHex) {
        this(L, M, bias, sigmOrTanh, linearDecoder, neuronsPerColumn, neuronsPerRow);
        if (orthOrHex) {
            double posX, posY;
            double stepX = 1.0;
            double stepY = 1.0;
            posY = 0.0;
            for (int dimY = 0; dimY < neuronsPerColumn; dimY++) {
                posX = 0.0;
                for (int dimX = 0; dimX < neuronsPerRow; dimX++) {
                    neuronPosition[dimY * neuronsPerRow + dimX][0] = posX;
                    neuronPosition[dimY * neuronsPerRow + dimX][1] = posY;
                    posX += stepX;
                }
                posY += stepY;
            }
        }
    }                          
    
    /*
    Estimation of the Gaussian distances (i.e. neighborhood parameters) between a
    neuron c and the rest of the neurons with respect to the given sigma value.
    On the network grid the closest neighboring neurons have a squared Euclidean
    distance of 1, the second closest neighboring neurons (diagonal) have a squared
    Euclidean distance of 3 (2 for the orthogonal), the third closest neighboring
    neurons have a squared Euclidean distance of 4 e.t.c.
    */
    double[] gaussianDistance(int c, double sigma) {
        double sumOfSquaredDifferences;
        double hde[] = new double[E];

        for (int e = 0; e < E; e++) {
            sumOfSquaredDifferences = 0.0;
            for (int dim = 0; dim < neuronPosition[0].length; dim++) {
                sumOfSquaredDifferences += (neuronPosition[c][dim] - neuronPosition[e][dim])*
                                           (neuronPosition[c][dim] - neuronPosition[e][dim]);
            }
            hde[e] = Math.exp(-sumOfSquaredDifferences / (2 * sigma * sigma));
        } 
        return hde;
    }
   
    /*
    Addition of Gaussian noise to every element of the input, nextGaussian returns
    values from the N(0,1) distribution in order to produce values belonging to
    the N(μ,σ^2) the usual equation is reversed:  Χ=Z*σ+μ
    */
    double[] isotropicGaussianNoise(double x[], double standardDeviation) {
       double x_bar[] = new double[x.length];
       Random noise = new Random();

       for (int l = 0; l < x.length; l++)    
           x_bar[l] = noise.nextGaussian() * standardDeviation + x[l];
       return x_bar;
    }
    
    /*
    Switches on/off (or at a desired value) a specific percentage of input
    elements. A percentage of 0% leaves unaffected all elements whereas
    a percentage of 100% sets all features equal to forcedValue.
    */
    double[] maskingNoise(double x[], double percentage, double forcedValue)
    {
        double x_bar[] = new double[x.length];
        Random noise = new Random();
        
        percentage /= 100.0;
        for (int l = 0; l < x.length; l++)
            if (noise.nextDouble() < percentage)
                x_bar[l] = forcedValue;
            else
                x_bar[l] = x[l];
        return x_bar;
    }
   
    /*
    Sigmoid function (0,1), s like.
    */
    double[] sigmoid(double z[]) {
       double y[] = new double[z.length];

       for (int i = 0; i<z.length; i++)
           y[i] = 1.0/(1.0+Math.exp(-z[i]));
       return y;
    }
   
    /*
    Derivative of the sigmoid function (0,0.25], bell like.
    */
    double[] sigmoidDerivative(double z[]) {
       double y[] = sigmoid(z);
       double dy[] = new double[z.length];

       for (int i = 0; i<z.length; i++)
           dy[i] = y[i]*(1.0-y[i]);
       return dy;
    }

    /*
    Hyperbolic tangent (-1,1), s like.
    */
    double[] tanh(double z[]) {
       double y[] = new double[z.length];

       for (int i = 0; i<z.length; i++)
           y[i] = Math.tanh(z[i]);
       return y;
    }
   
    /*
    Derivative of the hyperbolic tangent (0,1], bell like.
    */
    double[] tanhDerivative(double z[]) {
       double y[] = tanh(z);
       double dy[] = new double[z.length];

       for (int i = 0; i<z.length; i++)
           dy[i] = 1-y[i]*y[i];
       return dy;
    }
   
    /*
    Calculation of the transpose of weight matrix W, a copy is returned.
    */
    double[][] WT() {
        double transpose[][] = new double[L][M];
        
        for (int i = 0; i < M; i++)
            for (int j = 0; j < L; j++)
                transpose[j][i] = W[i][j];
        return transpose;
    }
    
    /*
    Calculation of the transpose of codebook matrix U, a copy is retunred.
    */
    double[][] UT() {
        double transpose[][] = new double[E][M];
        
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < E; j++) {
                transpose[j][i] = U[i][j];
            }
        }        
        
        return transpose;
    }
    
    double[] linearTransformation(double W[][], double x[]) {
        double z[] = new double[W.length];
       
        for (int m = 0; m < W.length; m++) {
            double sum = 0.0;
            for (int l = 0; l < x.length; l++)
                sum += W[m][l] * x[l];
            z[m] = sum;
        }
        return z;
    }
   
    double[] affineTransformation(double W[][], double x[], double b[]) {
        double z[] = new double[W.length];
       
        for (int m = 0; m < W.length; m++) {
            double sum = 0.0;
            for (int l = 0; l < x.length; l++)
                sum += W[m][l] * x[l];
            z[m] = sum + b[m];
        }  
        return z;
    }
    
    /*
    Referencing the paper is obligatory for understanding the present function.
    x is the actual pure sample, whereas x_bar is the noise corrupted sample.
    */
    void firstPhaseLearningStep(double x_bar[], double x[], double learningRate) {
        double z[];
        double df[];
        double y[];
        double z_hat[];
        double df_hat[];
        double x_hat[];
        double subtractionDerivative[] = new double[x.length];
        double summationMultiplication[] = new double[W.length];
        
        z = bias?affineTransformation(W, x_bar, b):linearTransformation(W, x_bar);
        df = sigmOrTanh?sigmoidDerivative(z):tanhDerivative(z);
        y = sigmOrTanh?sigmoid(z):tanh(z);
        z_hat = bias?affineTransformation(WT(), y, b_hat):linearTransformation(WT(), y);
        if (linearDecoder) {
            x_hat = z_hat;
            for (int j = 0; j<L; j++) 
                subtractionDerivative[j] = x[j]-x_hat[j];
        }
        else {
            df_hat = sigmOrTanh?sigmoidDerivative(z_hat):tanhDerivative(z_hat);
            x_hat = sigmOrTanh?sigmoid(z_hat):tanh(z_hat);   
            for (int j = 0; j<L; j++) 
                subtractionDerivative[j] = (x[j]-x_hat[j])*df_hat[j];
        }            
        for (int i = 0; i<M; i++) {
            summationMultiplication[i] = 0.0;
            for (int h = 0; h<L; h++)
                summationMultiplication[i] += subtractionDerivative[h]*W[i][h];
            summationMultiplication[i] *= df[i];
            for (int j = 0; j<L; j++) {
                W[i][j] += learningRate*(subtractionDerivative[j]*y[i]+
                           x_bar[j]*summationMultiplication[i]);
            }
        }
        if (bias) {
            for (int i = 0; i<M; i++) {
                b[i] += learningRate*summationMultiplication[i];
            }
            for (int j = 0; j<L; j++) {
                b_hat[j] += learningRate*subtractionDerivative[j];                
            }
        }     
    }
    
    /*
    Referencing the paper is obligatory for understanding the present function.
    x is the actual pure sample. IMPORTANT pay special attention to the
    bestMatchingNeuron() comments. Prefer low sigma values.
    */
    void secondPhaseLearningStep(double x[], double sigma, double aeLearningRate,
                                 double somLearningRate) {
        double z[];
        double df[];
        double y[];
        double hde[];
        double hce[];
        int c = -1;
        double weightedDistances;
        double minimumWeightedDistances = Double.POSITIVE_INFINITY;
        double squaredEuclidean[] = new double[neuronPosition.length];
        double summationMultiplication[] = new double[W.length];
        
        z = bias?affineTransformation(W, x, b):linearTransformation(W, x);
        df = sigmOrTanh?sigmoidDerivative(z):tanhDerivative(z);
        y = sigmOrTanh?sigmoid(z):tanh(z);
        for (int e = 0; e<E; e++) {
            squaredEuclidean[e] = 0.0;
            for (int m = 0; m<M; m++)
                squaredEuclidean[e] += (y[m]-U[m][e])*(y[m]-U[m][e]);
        }
        for (int d = 0; d<E; d++) {
            hde = gaussianDistance(d, sigma);
            weightedDistances = 0.0;
            for (int e = 0; e<E; e++)
                weightedDistances += hde[e]*squaredEuclidean[e];
            if (weightedDistances<=minimumWeightedDistances) {
                c = d;
                minimumWeightedDistances = weightedDistances;
            }                
        }
        hce = gaussianDistance(c, sigma);
        for (int i = 0; i<M; i++) {
            summationMultiplication[i] = 0.0;
            for (int e = 0; e<E; e++)
                summationMultiplication[i] += hce[e]*(U[i][e]-y[i]);
            summationMultiplication[i] *= df[i];
            for (int j = 0; j<L; j++)
                W[i][j] += aeLearningRate*x[j]*summationMultiplication[i];
        }
        if (bias)
            for (int i = 0; i<M; i++)
                b[i] += aeLearningRate*summationMultiplication[i];
        for (int i = 0; i<M; i++) 
            for (int e = 0; e<E; e++)
                U[i][e] += somLearningRate*hce[e]*(y[i]-U[i][e]);        
    }
    
    /*
    The index of the winner neuron is returned with respect to the employed
    storing scheme. In particular, positions are stored starting from the lower
    left neuron and proceeding in a left-to-right down-to-up fashion up to the
    last neuron which is stored in the upper right position. 
    Mathematically it cannot be proven that with different sigma values (used
    for the Gaussian distances) the winner neuron remains the same. In order to
    ensure this, initial spread values<0.2 need to be utilized, so that
    traditional winner neurons coincide with the Heskes-style ones.
    */
    int bestMatchingNeuron(double x[], double sigma) {
        double z[];
        double y[];
        double hde[];
        int c = -1;
        double weightedDistances;
        double minimumWeightedDistances = Double.POSITIVE_INFINITY;
        double squaredEuclidean[] = new double[neuronPosition.length];
                                                                                   
        z = bias?affineTransformation(W, x, b):linearTransformation(W, x);
        y = sigmOrTanh?sigmoid(z):tanh(z);        
        for (int e = 0; e<E; e++) {
            squaredEuclidean[e] = 0.0;
            for (int m = 0; m<M; m++)
                squaredEuclidean[e] += (y[m]-U[m][e])*(y[m]-U[m][e]);
        }
        for (int d = 0; d<E; d++) {
            hde = gaussianDistance(d, sigma);
            weightedDistances = 0.0;
            for (int e = 0; e<E; e++)
                weightedDistances += hde[e]*squaredEuclidean[e];
            if (weightedDistances<=minimumWeightedDistances) {
                c = d;
                minimumWeightedDistances = weightedDistances;
            }                
        }
        return c;
    }
    
    /*
    The winner (i.e. best-matching) neurons are detected by the typical
    way that incorporates the neighborhood function.
    */
    int[] bestMatchingNeuron(double samples[][], double sigma) {
        int c[] = new int[samples.length];
        
        for (int x = 0; x < samples.length; x++) {
            c[x] = bestMatchingNeuron(samples[x], sigma);
        }
        
        return c;
    }
    
    /*
    The winner (i.e. best-matching) neuron is detected by the traditional way.
    This translates to a winner neuron whose codebook vector is the closest one
    with respect to the input sample's hidden layer representation. In case
    of equidistant neurons the last (according to the storing scheme) is
    proclaimed winner.
    */
    int bestMatchingNeuron(double x[]) {
        double z[];
        double y[];
        int c = -1;
        double minimumDistance = Double.POSITIVE_INFINITY;
        double squaredEuclidean;
                                                                                          
        z = bias?affineTransformation(W, x, b):linearTransformation(W, x);
        y = sigmOrTanh?sigmoid(z):tanh(z);        
        for (int e = 0; e<E; e++) {
            squaredEuclidean = 0.0;
            for (int m = 0; m<M; m++)
                squaredEuclidean += (y[m]-U[m][e])*(y[m]-U[m][e]);
            if (squaredEuclidean<=minimumDistance) {
                c = e;
                minimumDistance = squaredEuclidean;
            }
        }
        return c;
    }
    
    /*
    The winner (i.e. best-matching) neurons are detected by the traditional way.
    This translates to winner neurons whose codebook vectors are the closest ones
    with respect to the input samples' hidden layer representations. In case
    of equidistant neurons the last (according to the storing scheme) is
    proclaimed winner.
    */
    int[] bestMatchingNeuron(double samples[][]) {
        int c[] = new int[samples.length];
        
        for (int x = 0; x < samples.length; x++) {
            c[x] = bestMatchingNeuron(samples[x]);
        }
        
        return c;
    }  
                     
    /*
    Calculation of the squared-error between input examples and their reconstructed
    instances. The returned value is averaged over the number of each sample's
    features and over the total number of available examples. Since this function
    should be used for normalized inputs, the provided featureLimits array must
    contain in its first row the minimum values and in its second row the maximum 
    value of all the features. It is important to note that the returned
    value corresponds to the raw (not normalized) value range of the attributes.
    In this way the comparison between the normalization and linear decoder
    approaches is made possible. The zeroOrMinusOne variable corresponds to the
    left value range margin of the normalized data.
    !Exactly because b_hat is not adjusted during the second training phase the
    values of squaredReconstructionError are NOT fully reliable!    
    */
    double squaredReconstructionError(double data[][], double featureLimits[][],
                                      boolean zeroOrMinusOne) {
        double averageJ;
        double x[];
        double z[];
        double y[];
        double z_hat[];
        double x_hat[];
        double reverseNormalization;
        
        averageJ = 0.0;
        for (int num = 0; num<data.length; num++) {
            x = data[num];
            z = bias?affineTransformation(W, x, b):linearTransformation(W, x);
            y = sigmOrTanh?sigmoid(z):tanh(z);
            z_hat = bias?affineTransformation(WT(), y, b_hat):linearTransformation(WT(), y);
            if (linearDecoder)
                x_hat = z_hat;
            else
                x_hat = sigmOrTanh?sigmoid(z_hat):tanh(z_hat);                                              
            for (int l = 0; l<L; l++) {
                if (zeroOrMinusOne)
                    reverseNormalization = (x[l]-x_hat[l])*
                                           (featureLimits[1][l]-featureLimits[0][l]);
                else
                    reverseNormalization = 0.5*(x[l]-x_hat[l])*                         //!!+!!
                                           (featureLimits[1][l]-featureLimits[0][l]);                                          
                averageJ += reverseNormalization*reverseNormalization;
            }
        }
        averageJ /= 2.0 * L * L * data.length;  //possibly the correct value needs L*L
        return averageJ;        
    }
    
    /*
    Calculation of the squared-error between input examples and their reconstructed
    instances. The returned value is averaged over the number of each sample's
    features and over the total number of available examples.
    !Exactly because b_hat is not adjusted during the second training phase the
    values of squaredReconstructionError are NOT fully reliable!    
    */
    double squaredReconstructionError(double data[][]) {
        double averageJ;
        double x[];
        double z[];
        double y[];
        double z_hat[];
        double x_hat[];
        
        averageJ = 0.0;
        for (int num = 0; num<data.length; num++) {
            x = data[num];
            z = bias?affineTransformation(W, x, b):linearTransformation(W, x);
            y = sigmOrTanh?sigmoid(z):tanh(z);
            z_hat = bias?affineTransformation(WT(), y, b_hat):linearTransformation(WT(), y);
            if (linearDecoder)
                x_hat = z_hat; 
            else
                x_hat = sigmOrTanh?sigmoid(z_hat):tanh(z_hat);
            for (int l = 0; l<L; l++)
                averageJ += (x[l]-x_hat[l])*(x[l]-x_hat[l]);
        }
        averageJ /= 2.0 * L * L * data.length;  //possibly the correct value needs L*L
        return averageJ;        
    }
    
    /*
    Numerically checking the computed derivatives/equations to make
    sure that the overall implementation is correct. Obviously, it is.
    */
    void gradientChecking(double epsilon) {
        double x[] = new double[L];  
        double z[];
        double df[];
        double y[];
        double z_hat[];
        double df_hat[] = new double[0];
        double x_hat[];
        double subtractionDerivative[] = new double[x.length];
        double summationMultiplication[] = new double[W.length];      
        double limit = 0.0;
        double minVal, maxVal;
        
        for (int l = 0; l < L; l++) {
            x[l] = -(L - l - 1) / (double) L;
        }

        int count = 1;
        for (int m = 0; m < M; m++) {
            for (int l = 0; l < L; l++) {
                W[m][l] = (count / (double) 10000) * (count / (double) 10000);
                count++;
            }
        }
        for (int m = 0; m < M; m++) {
            b[m] = 0.03 * (m + 1);     
        }
        for (int l = 0; l < L; l++) {
            b_hat[l] = -0.02 * (l + 1);
        }
        
        z = bias?affineTransformation(W, x, b):linearTransformation(W, x);
        df = sigmOrTanh?sigmoidDerivative(z):tanhDerivative(z);
        y = sigmOrTanh?sigmoid(z):tanh(z);
        z_hat = bias?affineTransformation(WT(), y, b_hat):linearTransformation(WT(), y);
        if (linearDecoder) {
            x_hat = z_hat;
            for (int j = 0; j<L; j++) 
                subtractionDerivative[j] = x[j]-x_hat[j];
        }
        else {
            df_hat = sigmOrTanh?sigmoidDerivative(z_hat):tanhDerivative(z_hat);
            x_hat = sigmOrTanh?sigmoid(z_hat):tanh(z_hat);   
            for (int j = 0; j<L; j++) 
                subtractionDerivative[j] = (x[j]-x_hat[j])*df_hat[j];
        }            
        for (int i = 0; i<M; i++) {
            summationMultiplication[i] = 0.0;
            for (int h = 0; h<L; h++)
                summationMultiplication[i] += subtractionDerivative[h]*W[i][h];
            summationMultiplication[i] *= df[i];
        }
        
        minVal = Double.POSITIVE_INFINITY;
        maxVal = Double.NEGATIVE_INFINITY;
        for (int m = 0; m < M; m++) {
            for (int l = 0; l < L; l++) {
                W[m][l] += epsilon;
                limit = jacobian(x, reconstruction(x));
                W[m][l] -= 2 * epsilon;     
                limit = (limit - jacobian(x, reconstruction(x))) / (2 * epsilon);
                W[m][l] += epsilon;      
                double partialDerivative = subtractionDerivative[l] * y[m] +
                    x[l] * summationMultiplication[m];
                //limit and -partialDerivative must coincide
                if (-partialDerivative - limit < minVal) {
                    minVal = -partialDerivative - limit;
                }
                else if (-partialDerivative - limit > maxVal) {
                    maxVal = -partialDerivative - limit;                    
                }
//                System.out.println(-partialDerivative - limit);
            }            
        }
        System.out.println(minVal + "\t" + maxVal);
        System.out.println();
        
        minVal = Double.POSITIVE_INFINITY;
        maxVal = Double.NEGATIVE_INFINITY;        
        for (int m = 0; m < M; m++) {
            b[m] += epsilon;
            limit = jacobian(x, reconstruction(x));
            b[m] -= 2 * epsilon;     
            limit = (limit - jacobian(x, reconstruction(x))) / (2 * epsilon);
            b[m] += epsilon;
            //limit and -summationMultiplication[m] must coincide
            if (-summationMultiplication[m] - limit < minVal) {
                minVal = -summationMultiplication[m] - limit;
            }
            else if (-summationMultiplication[m] - limit > maxVal) {
                maxVal = -summationMultiplication[m] - limit;                    
            }
//            System.out.println(-summationMultiplication[m] - limit);
        }     
        System.out.println(minVal + "\t" + maxVal);
        System.out.println();

        minVal = Double.POSITIVE_INFINITY;
        maxVal = Double.NEGATIVE_INFINITY;        
        for (int l = 0; l < L; l++) {
            b_hat[l] += epsilon;
            limit = jacobian(x, reconstruction(x));
            b_hat[l] -= 2 * epsilon;     
            limit = (limit - jacobian(x, reconstruction(x))) / (2 * epsilon);
            b_hat[l] += epsilon;
            //limit and -subtractionDerivative[l] must coincide
            if (-subtractionDerivative[l] - limit < minVal) {
                minVal = -subtractionDerivative[l] - limit;
            }
            else if (-subtractionDerivative[l] - limit > maxVal) {
                maxVal = -subtractionDerivative[l] - limit;                    
            }   
//            System.out.println(-subtractionDerivative[l] - limit);            
        }
        System.out.println(minVal + "\t" + maxVal);
        System.out.println();
    }
    
    double jacobian(double x[], double x_hat[]) {
        double energyCostError;
        
        energyCostError = 0.0;
        for (int l = 0; l < L; l++) {
            energyCostError += (x[l] - x_hat[l]) * (x[l] - x_hat[l]);
        }
        energyCostError *= 0.5;
        
        return energyCostError;
    }
                
    /*
    Self-explanatory, it returns the reconstructed value/approximation of the
    provided sample.
    */
    double[] reconstruction(double x[]) {
        double z[];
        double y[];
        double z_hat[];
        double x_hat[]; 
        
        z = bias ? affineTransformation(W, x, b) : linearTransformation(W, x);
        y = sigmOrTanh ? sigmoid(z) : tanh(z);
        z_hat = bias ? affineTransformation(WT(), y, b_hat) : linearTransformation(WT(), y);
        if (linearDecoder) {
            x_hat = z_hat; 
        }
        else {
            x_hat = sigmOrTanh ? sigmoid(z_hat) : tanh(z_hat);
        }
        
        return x_hat;
    }
    
    /*
    Self-explanatory, it returns the reconstructed value/approximation of the
    provided samples. As per usual each input sample is stored in a row.
    */
    double[][] reconstruction(double samples[][]) {
        double sampleReconstructions[][] = new double[samples.length][]; 
        
        for (int i = 0; i < samples.length; i++) {
            sampleReconstructions[i] = reconstruction(samples[i]);
        }
        
        return sampleReconstructions;
    }
    
    /*
    Characteristic paradigm for adjusting the parameters of the
    (denoising) autoencoder component/unit.
    */
    void trainDA(double samples[][]) {
        int steps = 100000;
        double initialLearningRate = 0.1;
        double finalLearningRate = 0.0001;
        double maskingPercentage = 20.0;
        double maskingValue = 0.0;
//        double noiseStandardDeviation = 2.0;
        double learningRate[];
        double x[];
        double x_bar[];
        
        DataManipulation.shuffle(samples, 1000*samples.length);   
        learningRate = FunctionValueSequence.linear(steps, initialLearningRate,
                                                    finalLearningRate);     
        
        System.out.println(squaredReconstructionError(samples));
        for (int loop = 0; loop < steps; loop++) {
            x = samples[loop%samples.length];  
            x_bar = maskingNoise(x, maskingPercentage, maskingValue);
//            x_bar = isotropicGaussianNoise(x, noiseStandardDeviation);
            firstPhaseLearningStep(x_bar, x, learningRate[loop]);
            if (loop % (steps/100) == 0)
                System.out.println(squaredReconstructionError(samples));
        }             
        System.out.println(squaredReconstructionError(samples));       
    }
    
    /*
    Characteristic paradigm of a complete DASOM learning procedure.
    */
    void trainDASOM(double samples[][])
    {
        int steps1stPhase = 100000;
        double initialLearningRate1stPhase = 0.5;
        double finalLearningRate1stPhase = 0.005;
        double maskingPercentage = 20.0;
        double maskingValue = 0.0;
//        double noiseStandardDeviation = 2.0;

        int steps2ndPhase = 200000;
        double initialLearningRate2ndPhase = 1.0; 
        double finalLearningRate2ndPhase = 0.01;
        double constantLearningRate2ndPhase = 0.0001;  
        double initialSigma = 0.6;
        double finalSigma = 0.2;          
        double learningRate[];
        double x[];
        double x_bar[];
        double sigma[];
        
        DataManipulation.shuffle(samples, 1000*samples.length); 
        
        /*
          1st Phase
        */  
        learningRate = FunctionValueSequence.linear(steps1stPhase,
                                                    initialLearningRate1stPhase,
                                                    finalLearningRate1stPhase);     
        for (int loop = 0; loop<steps1stPhase; loop++) {
            x = samples[loop%samples.length];  
            x_bar = maskingNoise(x, maskingPercentage, maskingValue);
//            x_bar = isotropicGaussianNoise(x, noiseStandardDeviation);
            firstPhaseLearningStep(x_bar, x, learningRate[loop]);
        }             
        System.out.println(squaredReconstructionError(samples));
        
        /*
          2nd Phase
        */
        learningRate = FunctionValueSequence.linear(steps2ndPhase,
                                                    initialLearningRate2ndPhase,
                                                    finalLearningRate2ndPhase);
        sigma = FunctionValueSequence.linear(steps2ndPhase, initialSigma,
                                             finalSigma);
        for (int loop = 0; loop<steps2ndPhase; loop++) {
            x = samples[loop%samples.length];
            secondPhaseLearningStep(x, sigma[loop], constantLearningRate2ndPhase, learningRate[loop]);
        }
        System.out.println(squaredReconstructionError(samples));
    }  
    
}
