import React, { useEffect, useState } from "react";
import axios from "axios";

const MultiversXTransactions = () => {
  const [transactions, setTransactions] = useState([]);
  const [inputTx, setInputTx] = useState("");
  const [analysisResult, setAnalysisResult] = useState("");

  const getRelativeTime = (timestamp) => {
    const now = Math.floor(Date.now() / 1000);
    const secondsAgo = now - timestamp;

    if (secondsAgo < 60) return `${secondsAgo} sec ago`;
    if (secondsAgo < 3600) return `${Math.floor(secondsAgo / 60)} min ago`;
    if (secondsAgo < 86400) return `${Math.floor(secondsAgo / 3600)} hr ago`;
    return `${Math.floor(secondsAgo / 86400)} days ago`;
  };

  const fetchTransactions = async () => {
    try {
      // Fetch transactions from Elrond API
      const response = await axios.get(
        "https://api.elrond.com/transactions?status=success&size=10"
      );
      let fetchedTransactions = response.data;

      // Prepare transactions for fraud analysis
      const formattedData = fetchedTransactions.map((tx) => ({
        txHash: tx.txHash,
        gasLimit: tx.gasLimit || 0,
        gasPrice: tx.gasPrice || 0,
        gasUsed: tx.gasUsed || 0,
        nonce: tx.nonce || 0,
        receiverShard: tx.receiverShard || 0,
        receiver:tx.receiver ||0,
        sender: tx.sender || 0,
        round: tx.round || 0,
        senderShard: tx.senderShard || 0,
        value: tx.value || 0,
        fee: tx.fee || 0,
        timestamp: tx.timestamp || 0,
      }));

      // Send transactions to Flask API for fraud detection
      const fraudResponse = await axios.post("http://127.0.0.1:5000/predict", {
        transactions: formattedData,
      });

      // Merge fraud predictions into transactions
      const predictions = fraudResponse.data.predictions || [];
      fetchedTransactions = fetchedTransactions.map((tx, index) => ({
        ...tx,
        isFraud: predictions[index].isFraud,
      }));
      console.log(predictions);
      setTransactions(fetchedTransactions);
    } catch (error) {
      console.error("Error fetching transactions:", error);
    }
  };

  useEffect(() => {
    fetchTransactions();
    const interval = setInterval(fetchTransactions, 4000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-gray-900 text-white flex flex-col items-center justify-start p-8 space-y-10 w-full">
      {/* LIVE TRANSACTIONS TABLE */}
      <div className="w-full bg-gray-800 p-6 rounded-md shadow-md">
        <h2 className="text-3xl font-bold mb-6 text-center">Live Transactions</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-sm text-left text-gray-400">
            <thead className="bg-gray-700 text-xs uppercase font-semibold text-gray-400">
              <tr>
                <th className="px-6 py-4">Txn Hash</th>
                <th className="px-6 py-4">Age</th>
                <th className="px-6 py-4">From</th>
                <th className="px-6 py-4">To</th>
                <th className="px-6 py-4">Method</th>
                <th className="px-6 py-4">Value</th>
                <th className="px-6 py-4">Suspicious</th>
              </tr>
            </thead>
            <tbody>
              {transactions.map((tx) => (
                <tr
                  key={tx.txHash}
                  className="border-b border-gray-700 hover:bg-gray-700 transition-colors"
                >
                  <td className="px-6 py-4 font-mono text-blue-400 truncate max-w-[300px]">
                    {tx.txHash}
                  </td>
                  <td className="px-6 py-4">
                    {tx.timestamp ? getRelativeTime(tx.timestamp) : "N/A"}
                  </td>
                  <td className="px-6 py-4 truncate max-w-[250px]">{tx.sender}</td>
                  <td className="px-6 py-4 truncate max-w-[250px]">{tx.receiver}</td>
                  <td className="px-6 py-4">
                    <span className="inline-block bg-green-500 text-green-900 rounded px-2 py-1 text-xs font-bold">
                      {tx.function || "N/A"}
                    </span>
                  </td>
                  <td className="px-6 py-4 text-yellow-400 font-semibold">{tx.value}</td>
                  <td className="px-6 py-4 text-center font-bold">
                    {tx.isFraud ? (
                      <span className="text-red-500">🚨 Yes</span>
                    ) : (
                      <span className="text-green-500">✔️ No</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default MultiversXTransactions;
