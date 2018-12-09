using System;
using System.Linq;
using ImageClassification.ImageData;
using System.IO;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime;
using static ImageClassification.Model.ConsoleHelpers;

namespace ImageClassification.Model
{
    public class ModelScorer
    {
        private readonly string dataLocation;
        private readonly string imagesFolder;
        private readonly string modelLocation;
        private readonly IHostEnvironment env;
        private ITransformer loadedModel;

        public ModelScorer(string dataLocation, string imagesFolder, string modelLocation)
        {
            this.dataLocation = dataLocation;
            this.imagesFolder = imagesFolder;
            this.modelLocation = modelLocation;
            env = new ConsoleEnvironment( seed: 1);
            LoadModel();
        }

        private void LoadModel()
        {
            ConsoleWriteHeader("Loading model");

            // Load the model
            using (var f = new FileStream(modelLocation, FileMode.Open))
            {
                loadedModel = TransformerChain.LoadFrom(env, f);
            }
            Console.WriteLine($"Model loaded: {modelLocation}");
        }
        public string ClassifyImage(string filename)
        {
            // Make prediction function (input = ImageNetData, output = ImageNetPrediction)
            var predictor = loadedModel.MakePredictionFunction<ImageNetData, ImageNetPrediction>(env);
            var testData = ImageNetData.ReadImage(filename,"---");

            ConsoleWriteHeader("Making classifications");
            // There is a bug (https://github.com/dotnet/machinelearning/issues/1138), 
            // that always buffers the response from the predictor
            // so we have to make a copy-by-value op everytime we get a response
            // from the predictor
            ImageNetPrediction data=  predictor.Predict(testData);
            
            return data.PredictedLabelValue + " " + data.Score.Max();
        }
    }
}
