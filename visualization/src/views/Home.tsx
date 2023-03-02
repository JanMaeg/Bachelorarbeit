import {
  Button,
  ButtonGroup,
  Card,
  CardBody,
  CardFooter,
  Container,
  Divider,
  Heading,
  SimpleGrid,
  Stack,
  Text,
} from "@chakra-ui/react";
import { Link } from "@tanstack/react-router";

const MethodCard = ({ title = "", description = "", link = "" }) => {
  return (
    <Card maxW="sm">
      <CardBody>
        <Stack mt="6" spacing="3">
          <Heading size="md">{title}</Heading>
          <Text>{description}</Text>
        </Stack>
      </CardBody>
      <Divider />
      <CardFooter>
        <ButtonGroup spacing="2">
          {/* @ts-ignore */}
          <Link to={link}>
            <Button variant="solid" colorScheme="blue">
              Go to view
            </Button>
          </Link>
        </ButtonGroup>
      </CardFooter>
    </Card>
  );
};

const Home = () => {
  return (
    <Container maxW="container.lg">
      <div className="home">
        <SimpleGrid columns={3} spacing={10}>
          <MethodCard
            title="Merge by string matching"
            description="Text is split. Splits are predicted separately. To merge a cluster
            with an already existing one, we count how often every token occurs
            in the already existing clusters. If one token appears only appears
            in one other cluster we count how often. At the end we merge the two
            clusters where the count is larger."
            link="/string-match"
          />
          <MethodCard
            title="Merge by overlapping split"
            description="Text is split whereby every split is overlapping by 2 sentences.
            Every split gets evaluated. Clusters are merged if two adjacent splits identify them."
            link="/window-merge"
          />
          <MethodCard
            title="Merge by embedding space"
            description="Text is split without overlapping.
            For every cluster the average embedding vector is calculated.
            For adjacent clusters cosine similarity is calculated.
            Merged if over a defined threshold."
            link="/embedding"
          />
          <MethodCard
            title="Merge by neural method"
            description="Text is split without overlapping."
            link="/neural"
          />
        </SimpleGrid>
      </div>
    </Container>
  );
};

export default Home;
