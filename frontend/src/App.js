import React, { useState } from "react";
import { Container, Typography } from "@mui/material";
import UploadArea from "./components/UploadArea";
import ResultViewer from "./components/ResultViewer";
import VideoUpload from "./components/VideoUpload";

function App() {
  const [result, setResult] = useState(null);
  const [file, setFile] = useState(null);

  const handleResult = (res, f) => {
    setResult(res);
    setFile(f);
  };

  return (
    <Container maxWidth="md" sx={{ py: 5 }}>
      <Typography variant="h4" align="center" gutterBottom>
        🚀 Détection Poubelle YOLOv9
      </Typography>
      <UploadArea onResult={handleResult} />
      <ResultViewer result={result} file={file} />
       <VideoUpload />
    </Container>
  );
}

export default App;
