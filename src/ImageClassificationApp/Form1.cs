using ImageClassification.Model;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace ImageClassificationApp
{
    public partial class Form1 : Form
    {
        Bitmap bmp;
        ModelScorer modelScorer;
        string pretrainedModel = @"pretrained\tensorflow_inception_graph.pb";
        string tagsTsv = "tags.tsv";
        string modelFile = "imageClassifier.zip";

        public Form1()
        {
            InitializeComponent();
        }
        private void CreateTsv(string imagePath)
        {
            string[] exts = { ".jpg", ".jpe", ".jpeg", ".bmp", ".png", ".gif" };

            var di = new DirectoryInfo(imagePath);
            var tags = di.EnumerateDirectories().Select(f => f.Name);
            List<string> data = new List<string>();
            foreach (string tag in tags)
            {
                string path = Path.Combine(imagePath, tag);
                var im = new DirectoryInfo(path);
                var files = im.EnumerateFiles("*.*", SearchOption.AllDirectories)
                    .Where(f => exts.Any(ext => f.FullName.ToLower().EndsWith(ext)))
                    .Select(f => f.FullName);
                foreach (string s in files)
                {
                    data.Add($"{s}\t{tag}");
                }
            }
            File.WriteAllLines(tagsTsv, data.ToArray());

        }
        private void LoadModel(string path)
        {
            var assetsPath = ModelHelpers.GetAssetsPath(path);
            var imagesFolder = ".";
            var imageClassifierZip = assetsPath;
            modelScorer = new ModelScorer(tagsTsv, imagesFolder, imageClassifierZip);
            textBox1.Text = "モデルを読み込みました。画像を Drag&Drop してください。";
        }
        private void Train(string path)
        {
            var assetsPath = ModelHelpers.GetAssetsPath(path);
            var imagesFolder = Path.Combine(assetsPath, "inputs", "data");
            var imageClassifierZip = Path.Combine(path, modelFile);

            try
            {
                var modelBuilder = new ModelBuilder(tagsTsv, imagesFolder, pretrainedModel, imageClassifierZip);
                modelBuilder.BuildAndTrain();
            }
            catch (Exception ex)
            {
                this.textBox1.Text = ex.Message;
            }
        }
        private void Predict(string filename)
        {
            try
            {
                // ファイルをロックしないようにするためコピーを作成
                using (Bitmap bitmap = new Bitmap(filename))
                {
                    bmp = new Bitmap(bitmap);
                    this.pictureBox1.Image = bmp;
                }
                Stopwatch watch = new Stopwatch();
                watch.Start();
                this.textBox1.Text = modelScorer.ClassifyImage(filename);
                watch.Stop();
                this.textBox1.Text += $" ({watch.ElapsedMilliseconds} ms)";
            }
            catch (Exception ex)
            {
                this.textBox1.Text = ex.Message;
            }
        }
        private void Form1_DragEnter(object sender, DragEventArgs e)
        {
            if (e.Data.GetDataPresent(DataFormats.FileDrop))
            {
                e.Effect = DragDropEffects.Copy;
            }
            else
            {
                e.Effect = DragDropEffects.None;
            }
        }

        private void Form1_DragDrop(object sender, DragEventArgs e)
        {
            string[] files = (string[])e.Data.GetData(DataFormats.FileDrop, false);
            Predict(files[0]);
        }

        private void buttonTrain_Click(object sender, EventArgs e)
        {
            if (folderBrowserDialog1.ShowDialog(this) == DialogResult.OK)
            {
                string path = folderBrowserDialog1.SelectedPath;
                CreateTsv(path);
                this.textBox1.Text = "学習しています。1分程度お待ちください。";
                this.Refresh();
                Train(path);
                LoadModel(Path.Combine(path, modelFile));
            }
        }

        private void buttonOpen_Click(object sender, EventArgs e)
        {
            openFileDialog1.FileName = modelFile;
            if (openFileDialog1.ShowDialog(this) == DialogResult.OK)
            {
                this.textBox1.Text = "モデルを読み込んでいます";
                this.Refresh();
                string path = openFileDialog1.FileName;
                LoadModel(path);
            }
        }
    }
}
