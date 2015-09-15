using System;
using System.Linq;

using Microsoft.VisualStudio.TestTools.UnitTesting;
using AIXI;

namespace AIXITests
{
    [TestClass]
    public class UtilsTests
    {
        [TestMethod]
        public void TestLogBase2()
        {
            
            Assert.AreEqual(3, Utils.LogBase2(8));
            Assert.AreEqual(2, Utils.LogBase2(7));
            Assert.AreEqual(2, Utils.LogBase2(4));
            Assert.AreEqual(1, Utils.LogBase2(2));
            Assert.AreEqual(0, Utils.LogBase2(1));
            Assert.AreEqual(-1, Utils.LogBase2(0));
            Assert.AreEqual(31, Utils.LogBase2(-1));
}
        [TestMethod]
        public void TestEncode()
        {
            int[] bits5 = { 1, 0, 1 };
            int[] bits2 = { 0, 1, 0 };
            int[] bits1 = { 0, 0, 1 };
            int[] bits0 = { 0, 0,0 };

            Assert.AreEqual(true, bits5.SequenceEqual(Utils.encode(5, 3)), "encoding of 5 failed");
            Assert.AreEqual(true, bits2.SequenceEqual(Utils.encode(2, 3)), "encoding of 2 failed");
            Assert.AreEqual(true, bits1.SequenceEqual(Utils.encode(1, 3)), "encoding of 1 failed");
            Assert.AreEqual(true, bits0.SequenceEqual(Utils.encode(0, 3)), "encoding of 0 failed");

        }

        [TestMethod]
        public void TestDecode() { 
            int[] bits = {1,0,1,1};
            Assert.AreEqual(13, Utils.decode(bits, 4));
            Assert.AreEqual(6, Utils.decode(bits, 3));
            Assert.AreEqual(3, Utils.decode(bits, 2));
            Assert.AreEqual(1, Utils.decode(bits, 1));
        }
    }
}
