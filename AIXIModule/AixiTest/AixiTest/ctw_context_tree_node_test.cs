using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using AIXI;

namespace AIXITests
{
    [TestClass]
    public class CtwContextTreeNodeTest
    {
        [TestMethod]
        public void TestCtwContextTreeNode()
        {

            var tree = new CTWContextTree(5);
            var n = new CTWContextTreeNode(tree);

            Assert.AreEqual(0.0, n.LogKt, 0.001);
            Assert.AreEqual(0.0, n.LogProbability, 0.001);
            Assert.AreEqual(0, n.SymbolCount(0));
            Assert.AreEqual(0, n.SymbolCount(1));
            Assert.AreEqual(0, n.NumberOf0S);
            Assert.AreEqual(0, n.NumberOf1S);
            Assert.AreEqual(tree, n.Tree);
            Assert.AreEqual(0, n.Visits());
            Assert.AreEqual(true, n.IsLeaf());
            Assert.AreEqual(1, n.Size());

            n.Update(1);
            n.Update(0);
            n.Update(0);
            n.Update(0);
            n.Update(1);

            Assert.AreEqual(-4.4465, n.LogKt, 0.001);
            Assert.AreEqual(-4.44656, n.LogProbability, 0.001);
            Assert.AreEqual(3, n.SymbolCount(0));
            Assert.AreEqual(2, n.SymbolCount(1));
            Assert.AreEqual(3, n.NumberOf0S);
            Assert.AreEqual(2, n.NumberOf1S);
            Assert.AreEqual(tree, n.Tree);
            Assert.AreEqual(5, n.Visits());
            Assert.AreEqual(true, n.IsLeaf());
            Assert.AreEqual(1, n.Size());


            n.Revert(1);

            Assert.AreEqual(-3.2425, n.LogKt, 0.001);
            Assert.AreEqual(-3.24259, n.LogProbability, 0.001);
            Assert.AreEqual(3, n.SymbolCount(0));
            Assert.AreEqual(1, n.SymbolCount(1));
            Assert.AreEqual(3, n.NumberOf0S);
            Assert.AreEqual(1, n.NumberOf1S);
            Assert.AreEqual(tree, n.Tree);
            Assert.AreEqual(4, n.Visits());
            Assert.AreEqual(true, n.IsLeaf());
            Assert.AreEqual(1, n.Size());

            //Todo:test non-leaf
        }
    }
}
