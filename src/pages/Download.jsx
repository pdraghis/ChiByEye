import React from "react";

const Download = () => (
  <section id="download">
    <h2>Download</h2>
    <p>To run the software, you need:</p>
    <ul>
      <li>Python 3</li>
      <li>XSPEC and pyxspec</li>
      <li>numpy, matplotlib, and PyQt5</li>
    </ul>
    <p>
      You can download the latest version from our
      <a
        href="https://github.com/pdraghis/ChiByEye.git"
        target="_blank"
        rel="noopener noreferrer"
      >
        {" "}
        GitHub repository
      </a>
      .
    </p>
  </section>
);

export default Download;
