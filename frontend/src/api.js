import axios from "axios";

const api = axios.create({
  baseURL: "http://localhost:5000", // ton Flask backend
});

export default api;
