using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using AIXI;

namespace AIXITests
{
    [TestClass]
    public class CTWContextTreeNodeTest
    {
        [TestMethod]
        public void TestCTWContextTreeNode()
        {

            var tree = new CTWContextTree(5);
            var n = new CTWContextTreeNode(tree);

            Assert.AreEqual(0.0, n.log_kt, 0.001);
            Assert.AreEqual(0.0, n.log_probability, 0.001);
            Assert.AreEqual(0, n.SymbolCount(0));
            Assert.AreEqual(0, n.SymbolCount(1));
            Assert.AreEqual(0, n.numberOf0s);
            Assert.AreEqual(0, n.numberOf1s);
            Assert.AreEqual(tree, n.tree);
            Assert.AreEqual(0, n.visits());
            Assert.AreEqual(true, n.IsLeaf());
            Assert.AreEqual(1, n.size());

            n.update(1);
            n.update(0);
            n.update(0);
            n.update(0);
            n.update(1);

            Assert.AreEqual(-4.4465, n.log_kt, 0.001);
            Assert.AreEqual(-4.44656, n.log_probability, 0.001);
            Assert.AreEqual(3, n.SymbolCount(0));
            Assert.AreEqual(2, n.SymbolCount(1));
            Assert.AreEqual(3, n.numberOf0s);
            Assert.AreEqual(2, n.numberOf1s);
            Assert.AreEqual(tree, n.tree);
            Assert.AreEqual(5, n.visits());
            Assert.AreEqual(true, n.IsLeaf());
            Assert.AreEqual(1, n.size());


            n.revert(1);

            Assert.AreEqual(-3.2425, n.log_kt, 0.001);
            Assert.AreEqual(-3.24259, n.log_probability, 0.001);
            Assert.AreEqual(3, n.SymbolCount(0));
            Assert.AreEqual(1, n.SymbolCount(1));
            Assert.AreEqual(3, n.numberOf0s);
            Assert.AreEqual(1, n.numberOf1s);
            Assert.AreEqual(tree, n.tree);
            Assert.AreEqual(4, n.visits());
            Assert.AreEqual(true, n.IsLeaf());
            Assert.AreEqual(1, n.size());

            //Todo:test non-leaf
        }
    }
}
