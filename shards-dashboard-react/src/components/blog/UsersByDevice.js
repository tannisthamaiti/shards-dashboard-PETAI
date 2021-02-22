import React from "react";
import PropTypes from "prop-types";
// import Placeholder from'./placeholder-image.png';
// import '../blog/ImagePosition.css';
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


const UserByDevice = ({ title, discussions }) => (
  <Card small className="h-100">
        <CardHeader className="border-bottom">
          <h6 className="m-0">{title}</h6>
        </CardHeader>
        <CardBody className="d-flex py-0">
          <Form className="quick-post-form">
        
        {/* Body */}
        <div class="ex1">
          <img src={discussions.image}  width ="500" alt="N/A"  object-fit="cover" class="center"/>
        </div>
		</Form>
        </CardBody>
        <CardFooter className="border-top">
          <Row>
            <Col>
              <FormSelect
                size="sm"
                value="last-week"
                style={{ maxWidth: "130px" }}
                onChange={() => {}}
              >
                <option value="last-week">Last Week</option>
                <option value="today">Today</option>
                <option value="last-month">Last Month</option>
                <option value="last-year">Last Year</option>
              </FormSelect>
            </Col>
            <Col className="text-right view-report">
              {/* eslint-disable-next-line */}
              <a href="#">View full report &rarr;</a>
            </Col>
          </Row>
        </CardFooter>
      </Card>
);

UserByDevice.propTypes = {
  /**
   * The component's title.
   */
  title: PropTypes.string,
  discussions: PropTypes.array
};

UserByDevice.defaultProps = {
  title: "Correlated Statigraphy per Well",
  discussions: [
    {
      image : require("../../images/placeholder-image.png"),
    }
  ]
};


export default UserByDevice;
