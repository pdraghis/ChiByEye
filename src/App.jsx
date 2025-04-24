import React from 'react';
import { HashRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Home from './pages/Home';
import About from './pages/About';
import Download from './pages/Download';
import Contact from './pages/Contact';
import './App.css';

const App = () => (
  <div className="app-root">
    <Router>
      <header>
        <div><strong>ChiByEye</strong></div>
        <Navbar />
      </header>
      <div className="container">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/about" element={<About />} />
          <Route path="/download" element={<Download />} />
          <Route path="/contact" element={<Contact />} />
        </Routes>
      </div>
      <footer>&copy; 2025 ChiByEye Project. All rights reserved.</footer>
    </Router>
  </div>
);

export default App;
