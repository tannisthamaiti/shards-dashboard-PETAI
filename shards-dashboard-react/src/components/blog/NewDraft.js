import React from "react";
import PropTypes from "prop-types";
import { ListGroupItem, Slider } from "shards-react";
import Tile3Card from './Tile3Card.js';

import {
  Row, 
  Col,
  Card,
  CardHeader,
  CardBody,
  CardFooter,
  Form,
  FormGroup,
  FormInput,
  FormTextarea,
  Button
} from "shards-react";


class NewDraft extends Component {

    state = {
      selectedFile: null
    };

    onFileChange = event => {
      this.setState({ selectedFile: event.target.files[0] });

    };
    onFileUpload = () => {
      console.log(this.state.selectedFile);
    };

    fileData = () => {

      if (this.state.selectedFile) {

        return (


<div >

              <img src={well} width ="500" alt="N/A"  object-fit="cover" class="center"/>
</div>

        );
      } else {
        return (
          <div>

          </div>
        );
      }
    };

    render() {

      return (
        <div class="w3-container" style={{ width: "600px" }}>
            <h3>
              Well Log Analysis
            </h3>
            <h4>
              Upload well log file in csv format.
            </h4>


            <div>
                <input type="file" onChange={this.onFileChange} />
                <button className="btn btn-success" onClick={this.onFileUpload}>
                  Upload
                </button>
            </div>
          {this.fileData()}
        </div>

      );
    }
  }



export default Case1;