using System;
using AIXI;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace UtilsTest
{
    [TestClass]
    public class TwoCTWCompare
    {
        [TestMethod]
        public void TestMethod1()
        {
            //here we make instances of CTWContextTreeFast and CTWContextTree and test if they behave in same way
            var ctf = new CTWContextTreeFast(9);
            var ct = new CTWContextTree(9);

            int[] input = { 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1 };
            ct.update_tree(input);
            ctf.update_tree(input);
            ct.revert_tree(4);
            ctf.revert_tree(4);
            int[] input2 = { 0, 0, 1 };
            ct.update_tree(input2);
            ctf.update_tree(input2);

            Assert.IsTrue(ctf.compare(ct));
        }
    }
}
