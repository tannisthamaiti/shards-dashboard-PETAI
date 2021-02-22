import React from "react";
import PropTypes from "prop-types";
import Placeholder from'./placeholder-image.png';
import '../blog/ImagePosition.css';
import {
  Card,
  CardHeader,
  CardBody,
  ListGroup,
  ListGroupItem,
  CardFooter,
  Row,
  Col,
  FormSelect
} from "shards-react";

import {Slider,Progress } from "shards-react";

import {
  Form,
  FormGroup,
  FormInput,
  FormTextarea,
  Field,
  Button,
} from "shards-react";


const TopReferrals = ({ title }) => (
  <Card small className="h-100">
    {/* Card Header */}
    <CardHeader className="border-bottom">
      <h6 className="m-0">{title}</h6>
    </CardHeader>

    <CardBody className="d-flex flex-column">
	<Row className="border-bottom py-2 bg-light">
	 <div>
      <FormInput size="sm" placeholder="Number of Iterations" className="mb-2" />
	  </div>
          </Row>
		 
      <Form className="quick-post-form">
        
        {/* Body */}
        <div class="ex1">
          <img src={Placeholder}  width ="500" alt="N/A"  object-fit="cover" class="center"/>
        </div>
		</Form>
		</CardBody>
		<CardFooter className="border-top">
        {/* Create Draft */}
        <div>
          <Button theme="accent" type="submit">
            Train ML Model
          </Button>
		  <Progress style={{ height: "5px" }} value={50} className="mb-3" />
     
		  <Button theme="accent" type="submit">
            Plot FMI Segments
          </Button>
        </div>
		</CardFooter>
  </Card>
);

TopReferrals.propTypes = {
  /**
   * The component's title.
   */
  title: PropTypes.string
};

TopReferrals.defaultProps = {
  title: "Feature Analysis FMI Logs"
};


export default TopReferrals;
