import React from "react";
import PropTypes from "prop-types";
import { ListGroupItem, Slider } from "shards-react";
import placeholder from './placeholder-image.png';
import './ImagePosition.css';

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


const NewDraft = ({ title }) => (
  <Card small className="h-100">
    {/* Card Header */}
    <CardHeader className="border-bottom">
      <h6 className="m-0">{title}</h6>
    </CardHeader>

    <CardBody className="d-flex flex-column">
	<p>Select Depth Range (ft/m):</p>
	<div className="mb-2 pb-1">
	<Slider
        connect
        start={[35, 65]}
        pips={{
          mode: "positions",
          values: [0, 25, 50, 75, 100],
          stepped: true,
          density: 5
        }}
        range={{ min: 0, max: 100 }}
      />   
	  </div>
      <Form className="quick-post-form">
        
         {/* Body */}
        <div>
          <img src={placeholder} width ="500" alt="N/A"  object-fit="cover" class="center"/>
        </div>
		</Form>
		</CardBody>
		 <CardFooter className="border-top">
        {/* Create Draft */}
        <FormGroup className="mb-0">
          <Button theme="accent" type="submit">
            Upload FMI Log
          </Button>
        </FormGroup>
      
    </CardFooter>
  </Card>
);

NewDraft.propTypes = {
  /**
   * The component's title.
   */
  title: PropTypes.string
};

NewDraft.defaultProps = {
  title: "Vizualize FMI Log"
};

export default NewDraft;
